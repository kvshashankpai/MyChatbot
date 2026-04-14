"""
model/transformer.py

HappyBot top-level module.
  - Owns the single shared Embedding (encoder input = decoder input = output projection)
  - Enforces weight tying: decoder.output_proj.weight ← embedding.weight
  - Exposes forward() for training and encode() / decode_step() for inference
"""

import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder


class HappyBot(nn.Module):

    def __init__(
        self,
        vocab_size: int       = 10_000,
        d_model: int          = 256,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int        = 2,
        d_ff: int             = 1024,
        max_len: int          = 512,
        dropout: float        = 0.1,
        num_emotion_classes: int   = 32,
        num_strategy_classes: int  = 8,
        pad_token_id: int     = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.vocab_size   = vocab_size
        # Store config so save_checkpoint can read them off the model
        self.d_model      = d_model
        self.d_ff         = d_ff
        self.num_heads    = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p    = dropout

        # ── Shared embedding ──────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[pad_token_id].zero_()

        # ── Encoder + Decoder ─────────────────────────────────────────────
        self.encoder = Encoder(
            embedding=self.embedding,
            d_model=d_model, num_layers=num_encoder_layers,
            num_heads=num_heads, d_ff=d_ff, max_len=max_len, dropout=dropout,
            num_emotion_classes=num_emotion_classes,
            num_strategy_classes=num_strategy_classes,
        )
        self.decoder = Decoder(
            embedding=self.embedding,
            d_model=d_model, num_layers=num_decoder_layers,
            num_heads=num_heads, d_ff=d_ff, max_len=max_len,
            dropout=dropout, vocab_size=vocab_size,
        )

        # ── Weight tying ──────────────────────────────────────────────────
        # output_proj.weight == embedding.weight (transposed during matmul)
        self.decoder.output_proj.weight = self.embedding.weight

    # ── Mask builders ─────────────────────────────────────────────────────

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, 1, 1, T)  True=real token"""
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    def make_tgt_pad_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, 1, 1, T)  True=real token"""
        return (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    # ── Training forward ──────────────────────────────────────────────────

    def forward(self, src, tgt, src_mask=None, tgt_pad_mask=None):
        if src_mask     is None: src_mask     = self.make_src_mask(src)
        if tgt_pad_mask is None: tgt_pad_mask = self.make_tgt_pad_mask(tgt)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc["memory"], src_mask, tgt_pad_mask)

        return {
            "logits":            dec["logits"],
            "emotion_logits":    enc["emotion_logits"],
            "strategy_logits":   enc["strategy_logits"],
            "cross_attn_weights": dec["cross_attn_weights"],
            "encoder_attn_weights": enc["attn_weights"],
        }

    # ── Inference helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, src, src_mask=None):
        """Run encoder once; cache result for autoregressive decoding."""
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        enc = self.encoder(src, src_mask)
        enc["src_mask"] = src_mask
        return enc

    @torch.no_grad()
    def decode_step(self, tgt_so_far, memory, src_mask):
        """
        Run the full decoder on tgt_so_far; return logits for the LAST position only.
        tgt_so_far : (B, T_gen)
        Returns    : (B, vocab_size)
        """
        dec = self.decoder(tgt_so_far, memory, src_mask, tgt_pad_mask=None)
        return dec["logits"][:, -1, :]   # next-token logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)