"""
model/attention.py

Scaled Dot-Product Attention + Multi-Head Attention.
Reused for encoder self-attention, decoder masked self-attention,
and decoder cross-attention.

MASK CONVENTION (enforced everywhere):
    True  = real token  → attend
    False = pad/future  → block (filled with -inf before softmax)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,            # (B, H, T_q, d_k)
    k: torch.Tensor,            # (B, H, T_k, d_k)
    v: torch.Tensor,            # (B, H, T_k, d_v)
    mask: torch.Tensor = None,  # broadcastable bool — True=attend, False=block
    dropout: nn.Dropout = None,
):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (B,H,T_q,T_k)

    if mask is not None:
        # Where mask is False → -inf so softmax gives 0 weight
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)   # guard fully-masked rows

    if dropout is not None:
        attn = dropout(attn)

    return torch.matmul(attn, v), attn  # (B,H,T_q,d_v), (B,H,T_q,T_k)


class MultiHeadAttention(nn.Module):
    """
    d_model=256, num_heads=2  →  d_k = d_v = 128 per head.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)

    def _split(self, x):   # (B,T,D) → (B,H,T,d_k)
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):   # (B,H,T,d_k) → (B,T,D)
        B, H, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(self, query, key, value, mask=None):
        q = self._split(self.W_q(query))
        k = self._split(self.W_k(key))
        v = self._split(self.W_v(value))

        out, attn_w = scaled_dot_product_attention(q, k, v, mask, self.attn_drop)
        return self.W_o(self._merge(out)), attn_w