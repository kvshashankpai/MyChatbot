"""
decoder.py — Transformer Decoder stack.
Includes:
  - DecoderLayer (masked MHA + cross-attention + FFN)
  - Decoder (embedding + PE + stack + output projection)
  
Key correctness invariants enforced here:
  1. Causal (upper-triangular) mask prevents position i from attending to i+1..T
  2. Cross-attention: Q from decoder hidden state, K and V from encoder memory
  3. Output projection weight is TIED to the input embedding matrix (set externally)
  4. Pre-norm (LayerNorm before each sublayer) for training stability from random init
"""

import math
import torch
import torch.nn as nn
from model.attention import MultiHeadAttention
from model.encoder import SinusoidalPositionalEncoding, FeedForward


class DecoderLayer(nn.Module):
    """
    Single decoder layer with three sublayers:
      1. Masked multi-head self-attention (causal)
      2. Cross-attention (Q from decoder, K/V from encoder memory)
      3. Position-wise FFN
    All with pre-norm + residual connections.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # (B, T_tgt, d_model)
        memory: torch.Tensor,      # (B, T_src, d_model) — encoder output
        tgt_mask: torch.Tensor,    # (B, 1, T_tgt, T_tgt) — causal mask (True=attend)
        src_mask: torch.Tensor,    # (B, 1, 1, T_src) — encoder padding mask (True=attend)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          x: (B, T_tgt, d_model)
          self_attn_weights: (B, H, T_tgt, T_tgt)
          cross_attn_weights: (B, H, T_tgt, T_src)  ← crucial for interpretability
        """
        # 1. Masked self-attention (decoder attends to its own past)
        residual = x
        x_norm = self.norm1(x)
        self_out, self_attn_w = self.self_attn(x_norm, x_norm, x_norm, mask=tgt_mask)
        x = residual + self.dropout(self_out)

        # 2. Cross-attention (decoder queries encoder memory)
        residual = x
        x_norm = self.norm2(x)
        cross_out, cross_attn_w = self.cross_attn(
            query=x_norm,
            key=memory,
            value=memory,
            mask=src_mask,
        )
        x = residual + self.dropout(cross_out)

        # 3. FFN
        residual = x
        x = residual + self.dropout(self.ffn(self.norm3(x)))

        return x, self_attn_w, cross_attn_w


class Decoder(nn.Module):
    """
    Full Decoder stack.
    Like the Encoder, this does NOT own the input embedding — it receives it
    from HappyBot to enforce weight tying with the encoder embedding.
    
    Weight tying: output_proj.weight == embedding.weight (set in HappyBot.__init__)
    This saves 256 * 10,000 = 2.56M parameters and improves perplexity.
    """

    def __init__(
        self,
        embedding: nn.Embedding,    # shared embedding (same object as encoder)
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        vocab_size: int = 10_000,
    ):
        super().__init__()
        self.embedding = embedding
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Output projection: d_model → vocab_size
        # Weight will be TIED to embedding.weight in HappyBot.__init__
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def _make_causal_mask(self, tgt_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates the causal (autoregressive) mask.
        Returns: (1, 1, T, T) boolean tensor.
        Lower-triangular = True (position i can attend to 0..i).
        Upper-triangular = False (blocked — future positions).
        """
        mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(
        self,
        tgt: torch.Tensor,         # (B, T_tgt) token ids
        memory: torch.Tensor,      # (B, T_src, d_model) encoder output
        src_mask: torch.Tensor,    # (B, 1, 1, T_src) encoder padding mask
        tgt_pad_mask: torch.Tensor = None,  # (B, 1, 1, T_tgt) decoder padding mask (optional)
    ) -> dict:
        """
        Returns:
          "logits": (B, T_tgt, vocab_size) — raw logits for next-token prediction
          "self_attn_weights": list of (B, H, T, T) per layer
          "cross_attn_weights": list of (B, H, T_tgt, T_src) per layer
        """
        T_tgt = tgt.size(1)
        device = tgt.device

        # Embedding + scale + positional encoding
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)

        # Causal mask: (1, 1, T_tgt, T_tgt)
        causal_mask = self._make_causal_mask(T_tgt, device)

        # Combine causal mask with decoder padding mask if provided
        if tgt_pad_mask is not None:
            # tgt_pad_mask: (B, 1, 1, T_tgt) — True=real token, False=pad
            # Expand pad mask to (B, 1, T_tgt, T_tgt) and AND with causal
            tgt_mask = causal_mask & tgt_pad_mask  # broadcasting handles shape
        else:
            tgt_mask = causal_mask

        self_attn_all = []
        cross_attn_all = []
        for layer in self.layers:
            x, self_w, cross_w = layer(x, memory, tgt_mask, src_mask)
            self_attn_all.append(self_w)
            cross_attn_all.append(cross_w)

        x = self.norm(x)  # Final LayerNorm

        logits = self.output_proj(x)  # (B, T_tgt, vocab_size)

        return {
            "logits": logits,
            "self_attn_weights": self_attn_all,
            "cross_attn_weights": cross_attn_all,
        }