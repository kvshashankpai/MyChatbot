"""
model/transformer.py — HappyBot Top-Level Module

This wraps Encoder + Decoder into a single nn.Module and:
  1. Owns and initialises the shared embedding (single source of truth).
  2. Handles all mask construction logic in one place.
  3. Implements the unified forward pass for training.
  4. Exposes encode() and decode_step() separately for inference.

Design decision: Encoder and Decoder share the same token embedding.
WHY: On a 10K vocabulary with d_model=256 the embedding matrix is
     10000×256 = 2.56M parameters. Sharing it across encoder input,
     decoder input, AND decoder output (weight tying) eliminates 5.12M
     redundant parameters — critical for generalisation on ~50K pairs.

FIX NOTES (NaN/Inf training crash):
  - shared_embedding now initialised with std=0.02 (GPT-style) and NO
    sqrt(d_model) forward scaling. The original code used std=d_model^-0.5
    (≈0.0625) and then scaled up 16× at runtime — net effect was std≈1.0
    which sounds fine, BUT the tied weight receives gradient from THREE
    paths simultaneously (enc_embed, dec_embed, dec_output_proj), so the
    effective gradient variance is 3× higher, pushing norms out of range
    during the very first backward pass.
  - Encoder is now constructed with shared_embedding= passed directly to
    its __init__ so it uses the same nn.Embedding object (not a post-hoc
    attribute override which could break if Encoder.__init__ ran its own
    nn.init calls on self.embedding after assignment).
  - nan_to_num guard on gen_logits REMOVED — it was hiding the upstream bug.
  - Added a forward-pass NaN diagnostic (debug_nans=True) that can be
    toggled at training time for fast fault isolation.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

from model.encoder import Encoder
from model.decoder import Decoder


class HappyBot(nn.Module):
    """
    Full Happy-Bot Encoder-Decoder Transformer.

    A single shared nn.Embedding is created here and passed into both
    Encoder and Decoder. The decoder's output projection is then tied to
    that same tensor (weight tying).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 2,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        num_emotion_classes: int = 32,
        num_strategy_classes: int = 8,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.d_model      = d_model
        self.vocab_size   = vocab_size

        # ── Shared embedding ───────────────────────────────────────────────
        # std=0.02: the de-facto standard for Transformer token embeddings
        # (GPT, BERT, etc.). With this init the embedding outputs are in
        # [-0.06, 0.06] before LayerNorm, which keeps attention scores sane
        # (max QK^T / sqrt(d_k) ≈ 0.06² × 128 / 11.3 ≈ 0.04) at step 0.
        self.shared_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        nn.init.normal_(self.shared_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.shared_embedding.weight[pad_token_id].fill_(0)

        # ── Encoder ────────────────────────────────────────────────────────
        # Pass shared_embedding to Encoder.__init__ so it assigns
        # self.embedding = shared_embedding BEFORE any further init runs.
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            num_emotion_classes=num_emotion_classes,
            num_strategy_classes=num_strategy_classes,
            shared_embedding=self.shared_embedding,
        )

        # ── Decoder ────────────────────────────────────────────────────────
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            shared_embedding=self.shared_embedding,
        )

        # ── Parameter count ────────────────────────────────────────────────
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"HappyBot initialized: {total:,} total params | {trainable:,} trainable")

    # ─────────────────────────────────────────────────────────────────────────
    # Mask helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _make_padding_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) bool mask — True where token == pad_token_id."""
        return ids == self.pad_token_id

    # ─────────────────────────────────────────────────────────────────────────
    # Training forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_ids: torch.Tensor,
        decoder_input: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None,
        debug_nans: bool = False,
    ) -> Dict[str, object]:
        """
        Full forward pass for training.

        Args:
            encoder_ids:    (B, S)  encoder input token IDs.
            decoder_input:  (B, T)  teacher-forced decoder input.
            encoder_mask:   (B, S)  pre-computed padding mask (True=pad); optional.
            decoder_mask:   (B, T)  pre-computed padding mask (True=pad); optional.
            debug_nans:     If True, print tensor stats at each sub-step.

        Returns dict:
            gen_logits:       (B, T, vocab_size)
            emotion_logits:   (B, num_emotion_classes)
            strategy_logits:  (B, num_strategy_classes)
            encoder_attn:     list of attention weight tensors
            cross_attn:       list of cross-attention weight tensors
        """
        enc_pad_mask = encoder_mask if encoder_mask is not None \
                       else self._make_padding_mask(encoder_ids)
        dec_pad_mask = decoder_mask if decoder_mask is not None \
                       else self._make_padding_mask(decoder_input)

        # ── Encoder ───────────────────────────────────────────────────────
        H_enc, emotion_logits, strategy_logits, enc_attn = self.encoder(
            src_ids=encoder_ids,
            src_key_padding_mask=enc_pad_mask,
        )

        if debug_nans:
            _check("H_enc", H_enc)
            _check("emotion_logits", emotion_logits)

        # ── Decoder (teacher forcing) ──────────────────────────────────────
        gen_logits, dec_self_attn, cross_attn = self.decoder(
            tgt_ids=decoder_input,
            memory=H_enc,
            tgt_key_padding_mask=dec_pad_mask,
            memory_key_padding_mask=enc_pad_mask,
        )

        if debug_nans:
            _check("gen_logits", gen_logits)

        return {
            "gen_logits":      gen_logits,
            "emotion_logits":  emotion_logits,
            "strategy_logits": strategy_logits,
            "encoder_attn":    enc_attn,
            "cross_attn":      cross_attn,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Inference: encode once
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(
        self,
        encoder_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run encoder and return cached outputs for autoregressive decoding.

        Returns:
            H_enc, emotion_logits, strategy_logits, enc_pad_mask
        """
        enc_pad_mask = self._make_padding_mask(encoder_ids)
        H_enc, emotion_logits, strategy_logits, _ = self.encoder(
            encoder_ids, src_key_padding_mask=enc_pad_mask
        )
        return H_enc, emotion_logits, strategy_logits, enc_pad_mask

    # ─────────────────────────────────────────────────────────────────────────
    # Inference: single decode step
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_step(
        self,
        decoder_ids: torch.Tensor,
        H_enc: torch.Tensor,
        enc_pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        One forward pass of the decoder for autoregressive generation.

        Returns:
            logits_last: (B, vocab_size) — logits for the NEXT token
            cross_attn:  list of cross-attention weights
        """
        dec_pad_mask = self._make_padding_mask(decoder_ids)

        logits, _, cross_attn = self.decoder(
            tgt_ids=decoder_ids,
            memory=H_enc,
            tgt_key_padding_mask=dec_pad_mask,
            memory_key_padding_mask=enc_pad_mask,
        )

        return logits[:, -1, :], cross_attn


# ─────────────────────────────────────────────────────────────────────────────
# Debug helper
# ─────────────────────────────────────────────────────────────────────────────

def _check(name: str, t: torch.Tensor) -> None:
    """Print tensor statistics. Call with debug_nans=True during debugging."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    print(f"  [{name}] shape={tuple(t.shape)} "
          f"min={t.min():.4f} max={t.max():.4f} "
          f"nan={has_nan} inf={has_inf}")
