"""
model/decoder.py — Transformer Decoder with Cross-Attention Bridge

Key spec constraints:
  • 4 decoder layers, same config as encoder (d_model=256, 2 heads)
  • Sub-layer 1: Masked causal self-attention
  • Sub-layer 2: Cross-attention (Q=decoder, K/V=H_enc from encoder)
  • Sub-layer 3: FFN
  • Output projection weight-tied to shared embedding matrix

FIX NOTES (NaN/Inf training crash):
  - sqrt(d_model) embedding scaling REMOVED. See encoder.py docstring for
    the full explanation. With shared weights and std=0.02 init, the 16x
    scaling was the direct cause of attention score overflow → NaN softmax.
  - nan_to_num on logits REMOVED — it was silently masking the upstream
    bug. Now that the root cause is fixed this guard is unnecessary noise.
  - FFN activation: ReLU → GELU (consistent with encoder change).
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from model.attention import MultiHeadAttention
from model.encoder import PositionalEncoding, FeedForwardNetwork


# ─────────────────────────────────────────────────────────────────────────────
# Causal (look-ahead) mask generation
# ─────────────────────────────────────────────────────────────────────────────

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Upper-triangular causal mask: position i cannot attend to positions j > i.

    Returns boolean tensor of shape (1, 1, seq_len, seq_len).
    True at positions (i, j) where j > i (future positions, blocked).
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
                      diagonal=1)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)


# ─────────────────────────────────────────────────────────────────────────────
# Single Decoder Layer
# ─────────────────────────────────────────────────────────────────────────────

class DecoderLayer(nn.Module):
    """
    One decoder layer. Three sub-layers:

      1. Masked Self-Attention — causal + padding mask; pre-norm.
      2. Cross-Attention — Q from decoder, K/V from encoder memory; pre-norm.
      3. FFN — same structure as encoder FFN; pre-norm.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 2,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()

        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1      = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2      = nn.LayerNorm(d_model)

        self.ffn        = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm3      = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # ── Sub-layer 1: Masked Self-Attention ────────────────────────────
        residual = tgt
        tgt_norm = self.norm1(tgt)

        combined_tgt_mask = self._combine_masks(
            tgt_mask, tgt_key_padding_mask, tgt_norm.size(1), tgt_norm.device
        )

        self_out, self_attn_w = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm, mask=combined_tgt_mask
        )
        tgt = residual + self.dropout(self_out)

        # ── Sub-layer 2: Cross-Attention ───────────────────────────────────
        residual = tgt
        tgt_norm = self.norm2(tgt)

        mem_attn_mask = None
        if memory_key_padding_mask is not None:
            mem_attn_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2)

        cross_out, cross_attn_w = self.cross_attn(
            query=tgt_norm,
            key=memory,
            value=memory,
            mask=mem_attn_mask,
        )
        tgt = residual + self.dropout(cross_out)

        # ── Sub-layer 3: FFN ───────────────────────────────────────────────
        residual = tgt
        tgt = residual + self.dropout(self.ffn(self.norm3(tgt)))

        return tgt, self_attn_w, cross_attn_w

    @staticmethod
    def _combine_masks(
        causal_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Merge causal (1,1,T,T) and padding (B,T) masks into (B,1,T,T).
        A position is blocked if EITHER mask says so.
        """
        if causal_mask is None:
            causal_mask = make_causal_mask(seq_len, device)

        if padding_mask is None:
            return causal_mask

        pad_expanded = padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
        return causal_mask | pad_expanded


# ─────────────────────────────────────────────────────────────────────────────
# Full Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """
    Full Transformer Decoder stack.

    Components:
      1. Token Embedding (shared with encoder via shared_embedding arg)
      2. Positional Encoding (sinusoidal, fixed)
      3. Stack of N DecoderLayers (N=4)
      4. Final LayerNorm
      5. Output projection — WEIGHT TIED to embedding matrix

    Weight Tying:
      output_proj.weight = embedding.weight
      Saves 10000×256 = 2.56M parameters and improves generalisation on
      small vocabularies by coupling the input and output token representations.
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
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Weight-tied output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            tgt_ids:                 Decoder input token IDs (B, T).
            memory:                  Encoder output H_enc (B, S, d_model).
            tgt_key_padding_mask:    Decoder padding mask (B, T), True = pad.
            memory_key_padding_mask: Encoder padding mask (B, S), True = pad.

        Returns:
            logits:             (B, T, vocab_size)
            self_attn_weights:  List[(B, h, T, T)] per layer
            cross_attn_weights: List[(B, h, T, S)] per layer
        """
        T = tgt_ids.size(1)

        # Embed tokens — NO sqrt(d_model) scaling.
        x = self.embedding(tgt_ids)   # (B, T, d_model)
        x = self.pos_encoding(x)

        causal_mask = make_causal_mask(T, tgt_ids.device)

        self_attn_weights_all  = []
        cross_attn_weights_all = []

        for layer in self.layers:
            x, self_w, cross_w = layer(
                tgt=x,
                memory=memory,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            self_attn_weights_all.append(self_w)
            cross_attn_weights_all.append(cross_w)

        x = self.norm(x)

        # Output projection — tied to embedding.weight
        logits = self.output_proj(x)   # (B, T, vocab_size)

        return logits, self_attn_weights_all, cross_attn_weights_all
