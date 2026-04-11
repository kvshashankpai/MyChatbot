"""
model/encoder.py — Transformer Encoder with Multi-Task NLU Heads

WHY the encoder exists separately from the decoder:
  The encoder's job is pure NLU (natural language understanding):
    • Full bidirectional attention over the entire conversation.
    • Extract a holistic [CLS] representation for classification.
    • Build a memory matrix H_enc that the decoder's cross-attention queries.

  These tasks are structurally different from the decoder's autoregressive
  generation role, hence the architectural separation.

Key spec constraints enforced:
  • 4 encoder layers (Section 5.2)
  • 2-head self-attention, d_model=256 (d_k=128 per head)
  • FFN: Linear(256→1024) → GELU → Linear(1024→256)
  • [CLS] token at position 0 accumulates classification signal
  • Emotion head: 2-layer MLP → num_emotion_classes logits
  • Strategy head: 2-layer MLP → 8 logits
  • Sinusoidal positional encoding (pre-computed, non-trainable)

FIX NOTES (NaN/Inf training crash):
  - Embedding init: use std=0.02 (GPT-style) instead of d_model^-0.5.
    The original code then multiplies by sqrt(d_model)=16 at forward time,
    so effective std was only 0.0625 — far too small and not the right
    pairing with the sqrt(d_model) scale factor.
  - Embedding sqrt(d_model) scaling: REMOVED from forward(). It is only
    correct when embedding weights are initialized with std=1/sqrt(d_model),
    and even then the benefit is marginal. With shared + tied weights the
    triple gradient path makes large embedding values catastrophic.
  - FFN activation: ReLU → GELU. GELU is smoother near 0 and does not
    produce dead neurons, reducing the chance of all-zero rows feeding into
    LayerNorm → NaN.
  - Removed output clamp on FFN (was hiding NaN rather than fixing it).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.attention import MultiHeadAttention


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encodings (Vaswani et al. 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    WHY sinusoidal (not learned):
      • Generalises to sequence lengths unseen during training.
      • Saves parameters — critical on small datasets.

    Registered as a buffer: included in state_dict but excluded from optimizer.
    """

    def __init__(self, d_model: int = 256, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, T, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Embedding tensor (B, T, d_model). Already scaled if needed.
        Returns:
            x + positional encoding, with dropout.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
# Feed-Forward Network
# ─────────────────────────────────────────────────────────────────────────────

class FeedForwardNetwork(nn.Module):
    """
    Position-wise FFN: Linear(d_model→d_ff) → GELU → Dropout → Linear(d_ff→d_model).

    d_ff = 1024 = 4 × d_model (standard 4× expansion).

    GELU replaces ReLU: smoother gradient near 0, avoids dead-neuron pathology
    that can cause all-zero rows → NaN in downstream LayerNorm.
    """

    def __init__(self, d_model: int = 256, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.act      = nn.GELU()

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.act(self.linear1(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Single Encoder Layer
# ─────────────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    One encoder layer with pre-norm residual connections.

    Operations:
      1. LayerNorm → Multi-Head Self-Attention → Dropout → Residual
      2. LayerNorm → FFN → Dropout → Residual

    Pre-norm is more stable than post-norm for training from scratch on
    small datasets (gradients stay consistent across depth).
    """

    def __init__(self, d_model: int = 256, num_heads: int = 2,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm + self-attention
        residual = x
        x_norm = self.norm1(x)

        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,S)

        attn_out, attn_weights = self.self_attn(x_norm, x_norm, x_norm, mask=attn_mask)
        x = residual + self.dropout(attn_out)

        # Pre-norm + FFN
        residual = x
        x = residual + self.dropout(self.ffn(self.norm2(x)))

        return x, attn_weights


# ─────────────────────────────────────────────────────────────────────────────
# Classification Heads
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """
    2-layer MLP classification head on the [CLS] token vector.

    Architecture: Linear(d_model, hidden) → GELU → Dropout → Linear(hidden, num_classes)
    """

    def __init__(self, d_model: int = 256, hidden: int = 128,
                 num_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, cls_vector: torch.Tensor) -> torch.Tensor:
        return self.layers(cls_vector)


# ─────────────────────────────────────────────────────────────────────────────
# Full Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Full Transformer Encoder stack.

    Components:
      1. Token Embedding (vocab_size → d_model)  — may be shared
      2. Positional Encoding (sinusoidal, fixed)
      3. Stack of N EncoderLayers (N=4)
      4. Final LayerNorm
      5. Emotion Head  (on [CLS] token)
      6. Strategy Head (on [CLS] token)

    Embedding note:
      The embedding weight is initialized with std=0.02 in HappyBot (top-level)
      and passed in via shared_embedding. The encoder does NOT rescale by
      sqrt(d_model) in its forward pass — that scaling was the primary source
      of NaN loss (see module docstring).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 2,
        num_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_emotion_classes: int = 32,
        num_strategy_classes: int = 8,
        shared_embedding: Optional[nn.Embedding] = None,
    ):
        super().__init__()

        self.d_model = d_model

        if shared_embedding is not None:
            self.embedding = shared_embedding
        else:
            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.embedding.weight[0].fill_(0)

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.emotion_head  = ClassificationHead(d_model, 128, num_emotion_classes, dropout)
        self.strategy_head = ClassificationHead(d_model, 128, num_strategy_classes, dropout)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """
        Args:
            src_ids:               Token ID tensor (B, S).
            src_key_padding_mask:  Padding mask (B, S), True = pad.

        Returns:
            H_enc:           (B, S, d_model)
            emotion_logits:  (B, num_emotion_classes)
            strategy_logits: (B, 8)
            attn_weights_list: list per layer
        """
        # Embed tokens — NO sqrt(d_model) scaling here.
        # With std=0.02 init the embedding magnitudes are already appropriate
        # for the downstream LayerNorm + attention pipeline.
        x = self.embedding(src_ids)          # (B, S, d_model)
        x = self.pos_encoding(x)

        attn_weights_list = []
        for layer in self.layers:
            x, attn_w = layer(x, src_key_padding_mask)
            attn_weights_list.append(attn_w)

        H_enc = self.norm(x)

        cls_vector      = H_enc[:, 0, :]
        emotion_logits  = self.emotion_head(cls_vector)
        strategy_logits = self.strategy_head(cls_vector)

        return H_enc, emotion_logits, strategy_logits, attn_weights_list
