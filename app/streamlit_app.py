
import os, json, math
import streamlit as st
import torch
import sentencepiece as spm
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# --- Resolve project root relative to this file (works on Streamlit Cloud & locally) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
TOKD = os.path.join(ROOT, "tokenizer")
CKPT = os.path.join(ROOT, "checkpoints")
SRC  = os.path.join(ROOT, "src")

SPM_MODEL = os.path.join(TOKD, "spm.model")
SPECIAL = json.load(open(os.path.join(TOKD, "special.json")))
PAD = SPECIAL.get("pad", 0)
BOS = SPECIAL.get("bos", 1)
EOS = SPECIAL.get("eos", 2)

# --- Load tokenizer ---
sp = spm.SentencePieceProcessor(); sp.load(SPM_MODEL)
vocab_size = sp.get_piece_size()

# --- Model (same as training) ---
import sys
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from src.transformer_model import TransformerSeq2Seq

@st.cache_resource
def load_model(ckpt_path: str, d_model:int=256, nhead:int=2, enc_layers:int=2, dec_layers:int=2, d_ff:int=1024, dropout:float=0.1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerSeq2Seq(vocab=vocab_size, d_model=d_model, n_heads=nhead,
                               num_encoder_layers=enc_layers, num_decoder_layers=dec_layers,
                               d_ff=d_ff, dropout=dropout, pad_id=PAD).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, device

def encode(text:str): return sp.encode(text, out_type=int)
def decode(ids:List[int]): return sp.decode(ids)

def build_prompt(user_text:str, emotion_opt:str="", situation_opt:str=""):
    emo = (emotion_opt or "").strip().lower()
    sit = (situation_opt or "").strip().lower()
    x = f"Emotion: {emo} | Situation: {sit} | Customer: {user_text.strip().lower()} | Agent:"
    return x

@torch.no_grad()
def greedy_decode(model, device, src_ids, max_len=64):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(PAD)
    ys = torch.full((1,1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len):
        logits, _ = model(src, ys, src_pad, ys.eq(PAD), need_xattn=False)
        nxt = torch.argmax(logits[:, -1, :].float(), dim=-1)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        if nxt.item() == EOS: break
    seq = ys[0].tolist()[1:]
    if EOS in seq: seq = seq[:seq.index(EOS)]
    return decode(seq)

@torch.no_grad()
def beam_decode(model, device, src_ids, beam_size=4, max_len=64):
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_pad = src.eq(PAD)
    beams = [(0.0, [BOS])]
    for _ in range(max_len):
        cands = []
        for score, seq in beams:
            if seq[-1] == EOS:
                cands.append((score, seq)); continue
            ys = torch.tensor([seq], dtype=torch.long, device=device)
            logits, _ = model(src, ys, src_pad, ys.eq(PAD), need_xattn=False)
            logp = torch.log_softmax(logits[0, -1, :].float(), dim=-1)
            topk = torch.topk(logp, beam_size)
            for prob, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                cands.append((score + prob, seq + [tok]))
        cands.sort(key=lambda x: x[0], reverse=True)
        beams = cands[:beam_size]
        if all(s[-1] == EOS for _, s in beams): break
    best = beams[0][1][1:]
    if EOS in best: best = best[:best.index(EOS)]
    return decode(best)

# ---------------- UI ----------------
st.set_page_config(page_title="Empathetic Bot", page_icon="💬", layout="centered")
st.title("Empathetic Dialogues Chatbot")
st.caption("Transformer (from scratch) — greedy/beam decoding")

ckpt_default = os.path.join(CKPT, "best_model.pt")
ckpt_path = st.sidebar.text_input("Checkpoint path", value=ckpt_default)
d_model = st.sidebar.selectbox("Embedding dim", [256,512], index=0)
nhead = st.sidebar.selectbox("Heads", [2,4,8], index=0)
enc_layers = st.sidebar.selectbox("Encoder layers", [2,3,4], index=0)
dec_layers = st.sidebar.selectbox("Decoder layers", [2,3,4], index=0)
dropout = st.sidebar.slider("Dropout", 0.0, 0.3, 0.1, 0.05)

model, device = load_model(ckpt_path, d_model, nhead, enc_layers, dec_layers, 1024, dropout)

if "history" not in st.session_state: st.session_state["history"] = []
SPECIAL = json.load(open(os.path.join(TOKD, "special.json")))
emotion_list = ["", *[e.replace("<emo_","").replace(">","").replace("_"," ") for e in SPECIAL.get("emotion_tokens", [])]]
emotion = st.selectbox("Emotion (optional)", emotion_list, index=0)
situation = st.text_input("Situation (optional)", "")
user_text = st.text_area("Your message", height=100, placeholder="Type what the customer would say...")

col1, col2 = st.columns([1,1])
with col1:
    decode_method = st.selectbox("Decoding", ["Greedy","Beam Search"], index=0)
with col2:
    beam_size = st.number_input("Beam size", 2, 10, 4, step=1)

max_len = st.slider("Max reply length", 16, 128, 64, step=8)

if st.button("Send", type="primary"):
    if not user_text.strip():
        st.warning("Enter a message first.")
    else:
        prompt = build_prompt(user_text, emotion, situation)
        src_ids = sp.encode(prompt, out_type=int)
        if decode_method == "Greedy":
            reply = greedy_decode(model, device, src_ids, max_len=max_len)
        else:
            reply = beam_decode(model, device, src_ids, beam_size=int(beam_size), max_len=max_len)
        st.session_state["history"].append({"user": user_text, "agent": reply})

for turn in st.session_state["history"]:
    st.markdown(f"**Customer:** {turn['user']}")
    st.markdown(f"**Agent:** {turn['agent']}")

st.divider()
st.caption("Tip: select an emotion that matches the input (e.g., afraid, angry, sad, sentimental).")
