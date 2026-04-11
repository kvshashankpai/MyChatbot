"""
transformer.py — HappyBot top-level module.

Responsibilities:
  - Owns the single shared embedding (encoder input = decoder input = decoder output projection)
  - Builds Encoder and Decoder, passes the shared embedding into both
  - Enforces weight tying: output_proj.weight = embedding.weight
  - Provides mask-building helpers (padding mask, causal mask is internal to Decoder)
  - Exposes forward() for training and encode()/decode() for inference
"""

import torch
import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder


class HappyBot(nn.Module):
    """
    HappyBot: Encoder-Decoder Transformer for psychologically grounded dialogue.

    Architecture overview:
      - Shared BPE embedding (vocab_size × d_model) used by encoder input,
        decoder input, AND decoder output projection (weight tying).
      - Encoder: 4 layers, 2-head self-attention, d_model=256, d_ff=1024
        + EmotionHead + StrategyHead on [CLS] token
      - Decoder: 4 layers, masked self-attn + cross-attention bridge + output projection

    Training forward pass:
      Input: (src, tgt_input, src_mask, emotion_labels, strategy_labels)
      Output: (gen_logits, emotion_logits, strategy_logits)
      Loss: L_gen + 0.3 * L_emotion + 0.3 * L_strategy

    Inference:
      1. encode(src) → cache memory + predict emotion/strategy
      2. decode(tgt_so_far, memory, src_mask) → next token logits
    """

    def __init__(
        self,
        vocab_size: int = 10_000,
        d_model: int = 256,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 2,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        num_emotion_classes: int = 32,
        num_strategy_classes: int = 8,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.vocab_size = vocab_size

        # ── Shared embedding ──────────────────────────────────────────────
        # This single embedding matrix is used in THREE places:
        #   1. Encoder input embedding
        #   2. Decoder input embedding
        #   3. Decoder output projection (weight tied, transposed)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model ** -0.5)
        # Zero out the padding embedding row
        with torch.no_grad():
            self.embedding.weight[pad_token_id].fill_(0.0)

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = Encoder(
            embedding=self.embedding,
            d_model=d_model,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            num_emotion_classes=num_emotion_classes,
            num_strategy_classes=num_strategy_classes,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = Decoder(
            embedding=self.embedding,
            d_model=d_model,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            vocab_size=vocab_size,
        )

        # ── Weight Tying ─────────────────────────────────────────────────
        # Tie decoder output_proj.weight to the shared embedding matrix.
        # This is the primary way to reduce overfitting on a small dataset.
        # The output projection computes: logits = hidden @ embedding.weight.T
        self.decoder.output_proj.weight = self.embedding.weight

    # ── Mask Builders ─────────────────────────────────────────────────────

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        Padding mask for encoder input.
        Returns: (B, 1, 1, T_src) — True where token is real (not PAD).
        The mask is broadcast to (B, H, T_q, T_k) inside attention.
        """
        return (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)

    def make_tgt_pad_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        Padding mask for decoder input (combined with causal mask inside Decoder).
        Returns: (B, 1, 1, T_tgt)
        """
        return (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(2)

    # ── Forward (Training) ────────────────────────────────────────────────

    def forward(
        self,
        src: torch.Tensor,            # (B, T_src)
        tgt: torch.Tensor,            # (B, T_tgt) — decoder INPUT (teacher-forced)
        src_mask: torch.Tensor = None,
        tgt_pad_mask: torch.Tensor = None,
    ) -> dict:
        """
        Full training forward pass.

        Returns dict with:
          "logits": (B, T_tgt, vocab_size) — generation logits
          "emotion_logits": (B, num_emotion_classes)
          "strategy_logits": (B, num_strategy_classes)
          "cross_attn_weights": list of (B, H, T_tgt, T_src) — for visualization
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        if tgt_pad_mask is None:
            tgt_pad_mask = self.make_tgt_pad_mask(tgt)

        # Encode
        enc_out = self.encoder(src, src_mask)
        memory = enc_out["memory"]

        # Decode
        dec_out = self.decoder(tgt, memory, src_mask, tgt_pad_mask)

        return {
            "logits": dec_out["logits"],
            "emotion_logits": enc_out["emotion_logits"],
            "strategy_logits": enc_out["strategy_logits"],
            "cross_attn_weights": dec_out["cross_attn_weights"],
            "encoder_attn_weights": enc_out["attn_weights"],
        }

    # ── Inference Helpers ─────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> dict:
        """
        Run encoder once and cache output for autoregressive decoding.
        Returns encoder output dict (memory, emotion_logits, strategy_logits, ...)
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        enc_out = self.encoder(src, src_mask)
        enc_out["src_mask"] = src_mask
        return enc_out

    @torch.no_grad()
    def decode_step(
        self,
        tgt_so_far: torch.Tensor,   # (B, T_generated_so_far)
        memory: torch.Tensor,        # (B, T_src, d_model) — cached encoder output
        src_mask: torch.Tensor,      # (B, 1, 1, T_src)
    ) -> torch.Tensor:
        """
        Run one full decoder forward with the tokens generated so far.
        Returns logits for the LAST position only: (B, vocab_size).
        """
        dec_out = self.decoder(tgt_so_far, memory, src_mask, tgt_pad_mask=None)
        # Take only the last position's logits for next-token prediction
        return dec_out["logits"][:, -1, :]  # (B, vocab_size)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)