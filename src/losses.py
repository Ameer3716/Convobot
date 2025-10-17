
import torch
import torch.nn.functional as F

def ce_loss(logits, targets, pad_id: int):
    V = logits.size(-1)
    logits_flat = logits.float().contiguous().view(-1, V)
    targets_flat = targets.contiguous().view(-1)
    loss_sum = F.cross_entropy(
        logits_flat, targets_flat,
        ignore_index=pad_id,
        reduction="sum"
    )
    n_tokens = targets_flat.ne(pad_id).sum().clamp_min(1)
    return loss_sum / n_tokens
