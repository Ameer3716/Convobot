
import math, torch
import torch.nn as nn
import torch.nn.functional as F

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, L, D]
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# --- Scaled Dot-Product Attention ---
def scaled_dot_attn(q, k, v, attn_mask=None, key_padding_mask=None, dropout=None):
    # q,k,v: [B, H, T, d]
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [B,H,T_q,T_k]
    if attn_mask is not None:
        scores = scores + attn_mask  # attn_mask should be -inf on masked positions, broadcastable
    if key_padding_mask is not None:
        # key_padding_mask: [B, T_k] True for PAD -> set -inf
        mask = key_padding_mask[:, None, None, :].to(scores.dtype)  # [B,1,1,T_k]
        scores = scores.masked_fill(mask.bool(), float('-inf'))
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    out = torch.matmul(attn, v)   # [B,H,T_q,d]
    return out, attn

# --- Multi-Head Attention ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: [B, T, D] -> [B, H, T, d]
        B, T, D = x.size()
        x = x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        return x

    def combine_heads(self, x):
        # x: [B, H, T, d] -> [B, T, D]
        B, H, T, d = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, H * d)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=False):
        # q,k,v: [B,T,D]
        Q = self.split_heads(self.w_q(q))
        K = self.split_heads(self.w_k(k))
        V = self.split_heads(self.w_v(v))
        # attn_mask expected shape broadcastable to [B,H,T_q,T_k]
        out, attn = scaled_dot_attn(Q, K, V, attn_mask, key_padding_mask, self.dropout)
        out = self.combine_heads(out)
        out = self.w_o(out)
        if need_weights:
            # return average over heads for visualization: [B,T_q,T_k]
            return out, attn.mean(dim=1)
        return out, None

# --- FeedForward ---
class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

# --- Encoder/Decoder Layers (Pre-LN) ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, src_key_padding_mask=None):
        # Self-Attn
        h = self.norm1(x)
        h,_ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(h)
        # FF
        h = self.norm2(x)
        h = self.ff(h)
        x = x + self.dropout(h)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, need_xattn=False):
        # Masked Self-Attn
        h = self.norm1(x)
        h,_ = self.self_attn(h, h, h, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = x + self.dropout(h)
        # Cross-Attn
        h = self.norm2(x)
        h, attn = self.cross_attn(h, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=need_xattn)
        x = x + self.dropout(h)
        # FF
        h = self.norm3(x)
        h = self.ff(h)
        x = x + self.dropout(h)
        return x, attn

# --- Encoder / Decoder stacks ---
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
    def forward(self, src, src_key_padding_mask=None):
        x = src
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, need_xattn=False):
        x = tgt
        last_attn = None
        for i, layer in enumerate(self.layers):
            need = need_xattn and (i == len(self.layers)-1)
            x, attn = layer(x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, need_xattn=need)
            if need: last_attn = attn
        return x, last_attn

# --- Full Transformer ---
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab:int, d_model:int=256, n_heads:int=2, 
                 num_encoder_layers:int=2, num_decoder_layers:int=2, 
                 d_ff:int=1024, dropout:float=0.1, pad_id:int=0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model)
        self.encoder = Encoder(d_model, n_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_decoder_layers, dropout)
        self.generator = nn.Linear(d_model, vocab)
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @staticmethod
    def causal_mask(T, device):
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, src_ids, tgt_in_ids, src_pad_mask=None, tgt_pad_mask=None, need_xattn=False):
        src = self.src_embed(src_ids) * math.sqrt(self.d_model)
        src = self.pos_enc(src)
        tgt = self.tgt_embed(tgt_in_ids) * math.sqrt(self.d_model)
        tgt = self.pos_enc(tgt)

        T = tgt_in_ids.size(1)
        tgt_mask = self.causal_mask(T, src_ids.device)

        memory = self.encoder(src, src_key_padding_mask=src_pad_mask)
        dec_out, xattn = self.decoder(
            tgt, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            need_xattn=need_xattn
        )
        logits = self.generator(dec_out)
        return logits, xattn
