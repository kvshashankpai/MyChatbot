"""
model/decoder.py

Transformer Decoder with:
  - PRE-NORM residual connections (same reason as encoder)
  - Causal (upper-triangular) mask for autoregressive self-attention
  - Cross-attention: Q from decoder, K/V from encoder memory (H_enc)
  - Weight-tied output projection (set by HappyBot after construction)
"""

import math
import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.encoder import SinusoidalPositionalEncoding, FeedForward


# ── Decoder Layer (PRE-NORM) ──────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    Three pre-norm sublayers:
        1. Masked self-attention  (causal)
        2. Cross-attention        (Q=decoder, K/V=encoder memory)
        3. FFN
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn         = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, src_mask):
        # 1. Masked self-attention
        n = self.norm1(x)
        sa_out, sa_w = self.self_attn(n, n, n, tgt_mask)
        x = x + self.drop(sa_out)

        # 2. Cross-attention  ← Q from decoder hidden state, K/V from encoder
        n = self.norm2(x)
        ca_out, ca_w = self.cross_attn(n, memory, memory, src_mask)
        x = x + self.drop(ca_out)

        # 3. FFN
        x = x + self.drop(self.ffn(self.norm3(x)))

        return x, sa_w, ca_w


# ── Full Decoder ──────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    embedding and output_proj.weight are set/tied by HappyBot — do NOT own them here.
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
        vocab_size: int = 10_000,
    ):
        super().__init__()
        self.embedding  = embedding
        self.scale      = math.sqrt(d_model)
        self.pos_enc    = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers     = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm        = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        # output_proj.weight is TIED to embedding.weight in HappyBot.__init__

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """
        Lower-triangular bool mask: True = can attend.
        Shape (1, 1, T, T) — broadcast over batch and heads.
        """
        return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool)
                          ).unsqueeze(0).unsqueeze(0)

    def forward(self, tgt, memory, src_mask, tgt_pad_mask=None):
        """
        tgt          : (B, T_tgt)            decoder input token ids
        memory       : (B, T_src, D)         encoder output
        src_mask     : (B, 1, 1, T_src)      True=real encoder token
        tgt_pad_mask : (B, 1, 1, T_tgt)      True=real decoder token  (optional)

        Returns dict: logits, self_attn_weights, cross_attn_weights
        """
        T = tgt.size(1)
        x = self.pos_enc(self.embedding(tgt) * self.scale)

        # Combine causal mask with optional padding mask
        causal = self._causal_mask(T, tgt.device)          # (1,1,T,T)
        tgt_mask = causal & tgt_pad_mask if tgt_pad_mask is not None else causal

        sa_weights, ca_weights = [], []
        for layer in self.layers:
            x, sa_w, ca_w = layer(x, memory, tgt_mask, src_mask)
            sa_weights.append(sa_w)
            ca_weights.append(ca_w)

        x = self.norm(x)
        return {
            "logits":             self.output_proj(x),   # (B, T, V)
            "self_attn_weights":  sa_weights,
            "cross_attn_weights": ca_weights,
        }