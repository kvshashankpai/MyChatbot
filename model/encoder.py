"""
model/encoder.py

Transformer Encoder with:
  - Sinusoidal positional encoding
  - PRE-NORM layers  (LayerNorm BEFORE each sublayer, not after)
  - Embedding scaled by √d_model before PE addition
  - EmotionHead + StrategyHead on the [CLS] token (position 0)

WHY PRE-NORM:
    Post-norm (LayerNorm after residual) requires the sublayer outputs to be
    small before normalization can help.  With random init they are NOT small,
    so gradients vanish in the first few epochs.  Pre-norm puts a clean signal
    into every sublayer from the very first step, making training from scratch
    stable without any tricks.
"""

import math
import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


# ── Positional Encoding ───────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)       # (L,1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Feed-Forward ──────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ── Encoder Layer (PRE-NORM) ──────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Pre-norm residual block:
        x = x + Dropout( MHA( LayerNorm(x) ) )
        x = x + Dropout( FFN( LayerNorm(x) ) )
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # 1. Self-attention (pre-norm)
        x = x + self.drop(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)[0])
        # 2. FFN (pre-norm)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ── NLU Classification Heads ──────────────────────────────────────────────────

class ClassificationHead(nn.Module):
    """Generic 2-layer MLP head operating on the [CLS] vector."""

    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cls_vec):   # (B, D) → (B, num_classes)
        return self.net(cls_vec)


# ── Full Encoder ──────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    embedding is passed in (shared with decoder) — do NOT create it here.
    Embedding is scaled by √d_model BEFORE adding positional encoding.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        num_emotion_classes: int = 32,
        num_strategy_classes: int = 8,
    ):
        super().__init__()
        self.embedding  = embedding
        self.scale      = math.sqrt(d_model)   # ← critical: must multiply before PE
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers     = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm           = nn.LayerNorm(d_model)   # final norm after all layers
        self.emotion_head   = ClassificationHead(d_model, num_emotion_classes, dropout)
        self.strategy_head  = ClassificationHead(d_model, num_strategy_classes, dropout)

    def forward(self, src, src_mask):
        """
        src      : (B, T)            token ids
        src_mask : (B, 1, 1, T)      True = real token, False = pad
        Returns dict with memory, emotion_logits, strategy_logits, attn_weights.
        """
        x = self.pos_enc(self.embedding(src) * self.scale)  # (B, T, D)

        attn_weights = []
        for layer in self.layers:
            # Forward through layer — we need the attn weights too, so call self_attn directly
            normed = layer.norm1(x)
            attn_out, aw = layer.self_attn(normed, normed, normed, src_mask)
            x = x + layer.drop(attn_out)
            x = x + layer.drop(layer.ffn(layer.norm2(x)))
            attn_weights.append(aw)

        x = self.norm(x)

        cls = x[:, 0, :]   # [CLS] token is always at position 0
        return {
            "memory":           x,
            "cls_vector":       cls,
            "emotion_logits":   self.emotion_head(cls),
            "strategy_logits":  self.strategy_head(cls),
            "attn_weights":     attn_weights,
        }