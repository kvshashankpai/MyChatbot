"""
attention.py — Scaled Dot-Product Attention + Multi-Head Attention
Reused identically for:
  (1) Encoder self-attention (bidirectional, padding mask only)
  (2) Decoder masked self-attention (causal + padding mask)
  (3) Decoder cross-attention (Q from decoder, K/V from encoder memory)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,          # (B, H, T_q, d_k)
    k: torch.Tensor,          # (B, H, T_k, d_k)
    v: torch.Tensor,          # (B, H, T_k, d_v)
    mask: torch.Tensor = None,  # (B, 1, T_q, T_k) or (B, 1, 1, T_k)  — True = keep, False = mask out
    dropout: nn.Dropout = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core attention computation. Returns (output, attention_weights).
    mask convention: True = ATTEND, False = IGNORE (masked out with -inf).
    This is the most common source of bugs — we enforce it here strictly.
    """
    d_k = q.size(-1)
    # Scores: (B, H, T_q, T_k)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Where mask is False (i.e. should be masked OUT), set to -inf
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn_weights = F.softmax(scores, dim=-1)

    # After softmax, -inf positions become 0 (nan guard for all-masked rows)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, v)  # (B, H, T_q, d_v)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.
    d_model=256, num_heads=2  →  d_k = d_v = 128 per head.
    No separate head-splitting complexity: we use reshape + transpose.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 128

        # Fused Q, K, V projections for efficiency
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) → (B, H, T, d_k)"""
        B, T, _ = x.size()
        x = x.view(B, T, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (B, H, T, d_k)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, d_k) → (B, T, d_model)"""
        B, H, T, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(
        self,
        query: torch.Tensor,    # (B, T_q, d_model)
        key: torch.Tensor,      # (B, T_k, d_model)
        value: torch.Tensor,    # (B, T_k, d_model)
        mask: torch.Tensor = None,  # (B, 1, T_q, T_k) — True=attend, False=ignore
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split_heads(self.W_q(query))   # (B, H, T_q, d_k)
        k = self._split_heads(self.W_k(key))     # (B, H, T_k, d_k)
        v = self._split_heads(self.W_v(value))   # (B, H, T_k, d_k)

        attn_out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout
        )

        # Merge heads and project back
        merged = self._merge_heads(attn_out)     # (B, T_q, d_model)
        output = self.W_o(merged)
        return output, attn_weights  # attn_weights: (B, H, T_q, T_k)