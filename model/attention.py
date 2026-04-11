"""
model/attention.py — Scaled Dot-Product and Multi-Head Attention

WHY this is a standalone module:
  The same MultiHeadAttention class is reused for THREE purposes:
    1. Encoder self-attention (bidirectional, padding mask only)
    2. Decoder masked self-attention (causal + padding mask)
    3. Decoder cross-attention (Q from decoder, K/V from encoder memory)

  Centralising avoids code duplication and ensures all three share identical
  weight shapes, making weight-loading and debugging straightforward.

Spec constraints enforced here:
  • d_model = 256, num_heads = 2  →  d_k = d_v = 128 per head
  • Scaling factor = sqrt(128) ≈ 11.31 (prevents softmax saturation)
  • Dropout applied to attention weights (post-softmax)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,    # (B, h, T_q, d_k)
    key:   torch.Tensor,    # (B, h, T_k, d_k)
    value: torch.Tensor,    # (B, h, T_k, d_k)
    mask:  Optional[torch.Tensor] = None,   # broadcast-compatible boolean mask
    dropout: Optional[nn.Dropout] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention as defined in 'Attention Is All You Need'.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    Mask convention:
      mask = True at positions that should be IGNORED (set to -inf before softmax).
      This covers both padding positions and causal (future) positions.

    Args:
        query, key, value: Attention tensors.
        mask:  Boolean tensor. True = block this position.
        dropout: Optional dropout applied to attention weights.

    Returns:
        Tuple of (attended_values, attention_weights).
        attention_weights shape: (B, h, T_q, T_k) — used for visualization.
    """
    d_k = query.size(-1)
    # Scale prevents vanishing gradients in softmax with large d_k
    scale = math.sqrt(d_k)

    # Compute raw attention scores: (B, h, T_q, T_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    # Apply mask: replace masked positions with large negative value
    # so they become ~0 after softmax
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    # Softmax over key dimension
    attn_weights = F.softmax(scores, dim=-1)

    # NaN guard: if entire row was masked → softmax produces NaN → replace with 0
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # Weighted sum of values: (B, h, T_q, d_v)
    output = torch.matmul(attn_weights, value)
    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Spec constraints (Section 5.1):
      d_model = 256, num_heads = 2  →  d_k = d_v = d_model // num_heads = 128

    The same module handles:
      • Self-attention:  query=key=value=x
      • Cross-attention: query=decoder_hidden, key=value=encoder_memory

    Args:
        d_model:   Model dimensionality (256).
        num_heads: Number of parallel attention heads (2).
        dropout:   Dropout rate on attention weights (0.1 per spec).
    """

    def __init__(self, d_model: int = 256, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads   # 128 per head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization for all projection matrices."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape (B, T, d_model) → (B, num_heads, T, d_k)."""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse split_heads: (B, num_heads, T, d_k) → (B, T, d_model)."""
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        query: torch.Tensor,                    # (B, T_q, d_model)
        key:   torch.Tensor,                    # (B, T_k, d_model)
        value: torch.Tensor,                    # (B, T_k, d_model)
        mask:  Optional[torch.Tensor] = None,   # True = blocked position
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            output:        (B, T_q, d_model)
            attn_weights:  (B, num_heads, T_q, T_k)  — for visualization
        """
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_out, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.attn_dropout
        )

        output = self.W_o(self.combine_heads(attn_out))

        return output, attn_weights
