"""
inference.py — Autoregressive Inference Engine

Implements the full inference pipeline (Section 7, Step 8):
  1. encode()  — tokenize input, run encoder, cache H_enc
  2. Predict strategy from argmax of Strategy Head logits
  3. Seed decoder with [BOS][strategy_token]
  4. Autoregressive decode() with:
       - Temperature scaling (T=0.85)
       - Nucleus sampling p=0.9 (default)
       - Beam search (width=4) for 'suggestion' and 'information' strategies
  5. Return strategy label + generated text

Run command:
  python inference.py \
    --checkpoint checkpoints/phase2/best_model.pt \
    --tokenizer_dir data/tokenizer \
    --input "I've been feeling so overwhelmed at work lately, I can't sleep."
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import HappyBotTokenizer, ID_TO_STRATEGY, NUM_STRATEGY_CLASSES
from model.transformer import HappyBot
from utils import load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Sampling helpers
# ─────────────────────────────────────────────────────────────────────────────

def top_p_sampling(
    logits: torch.Tensor,   # (1, vocab_size)
    temperature: float = 0.85,
    top_p: float = 0.9,
) -> int:
    """
    Temperature-scaled nucleus (top-p) sampling.

    Steps:
      1. Divide logits by temperature (T < 1 sharpens, T > 1 flattens).
      2. Sort vocabulary by descending probability.
      3. Find the smallest set of tokens whose cumulative probability >= top_p.
         This is the "nucleus" — all other tokens are zeroed out.
      4. Sample from the renormalised nucleus distribution.

    WHY T=0.85 and p=0.9 (Section 9 spec justification):
      T=0.85 sharpens the distribution slightly without collapsing to greedy
      decoding. p=0.9 creates a high-quality nucleus that eliminates low-prob
      vocabulary tail while preserving enough diversity for empathetic variety.
    """
    # Temperature scaling
    scaled_logits = logits / temperature

    # Sort by probability (descending)
    probs  = F.softmax(scaled_logits, dim=-1).squeeze(0)  # (V,)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)

    # Cumulative probability mask
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # Remove tokens beyond nucleus (where cumulative > top_p)
    # Keep at least 1 token (the top-1) to avoid empty nucleus
    nucleus_mask = cumulative - sorted_probs > top_p
    sorted_probs[nucleus_mask] = 0.0

    # Renormalise nucleus
    sorted_probs /= sorted_probs.sum()

    # Sample from nucleus
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token_id = sorted_idx[sampled_sorted_idx].item()

    return next_token_id


def beam_search(
    model: HappyBot,
    H_enc: torch.Tensor,
    enc_pad_mask: torch.Tensor,
    seed_ids: List[int],
    eos_id: int,
    pad_id: int,
    max_length: int = 128,
    beam_width: int = 4,
    device: torch.device = torch.device("cpu"),
) -> List[int]:
    """
    Beam search decoding for 'suggestion' and 'information' strategies.

    WHY beam search for these strategies (Section 7, Step 8):
      Action-oriented responses (suggestions, information) must be coherent,
      complete, and non-ambiguous. Nucleus sampling can produce incoherent
      fragments. Beam search maximises the total log-probability of the
      sequence, producing well-formed, complete actionable advice.

    Returns:
        List of token IDs for the best beam sequence.
    """
    # Each beam: (log_prob, token_ids_list)
    beams: List[Tuple[float, List[int]]] = [(0.0, seed_ids[:])]
    completed: List[Tuple[float, List[int]]] = []

    for step in range(max_length):
        if not beams:
            break

        all_candidates: List[Tuple[float, List[int]]] = []

        for log_prob, seq in beams:
            if seq[-1] == eos_id:
                completed.append((log_prob, seq))
                continue

            # Decoder forward for current beam
            dec_ids = torch.tensor([seq], dtype=torch.long, device=device)
            logits_last, _ = model.decode_step(dec_ids, H_enc, enc_pad_mask)
            # logits_last: (1, vocab_size)

            log_probs_step = F.log_softmax(logits_last.squeeze(0), dim=-1)  # (V,)
            top_log_probs, top_ids = log_probs_step.topk(beam_width)

            for lp, tok_id in zip(top_log_probs.tolist(), top_ids.tolist()):
                all_candidates.append((log_prob + lp, seq + [tok_id]))

        # Keep top beam_width candidates (score normalised by length)
        all_candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
        beams = all_candidates[:beam_width]

        # Check if all beams have terminated
        if all(s[-1] == eos_id for _, s in beams):
            completed.extend(beams)
            break

    completed.extend(beams)

    if not completed:
        return seed_ids

    # Return sequence with highest length-normalised log-probability
    best = max(completed, key=lambda x: x[0] / max(1, len(x[1])))
    return best[1]


# ─────────────────────────────────────────────────────────────────────────────
# Main inference class
# ─────────────────────────────────────────────────────────────────────────────

class HappyBotInference:
    """
    High-level inference wrapper.

    Usage:
        engine = HappyBotInference.from_checkpoint(
            checkpoint_path, tokenizer_dir, device
        )
        result = engine.generate("I feel lost and hopeless.")
        print(result["strategy"])    # e.g., "reflection"
        print(result["response"])    # generated therapeutic response
    """

    # Strategies that use beam search instead of nucleus sampling (Section 7 Step 8)
    BEAM_STRATEGIES = {"suggestion", "information"}

    def __init__(
        self,
        model: HappyBot,
        tokenizer: HappyBotTokenizer,
        device: torch.device,
        temperature: float = 0.85,
        top_p: float = 0.90,
        beam_width: int = 4,
        max_src_len: int = 512,
        max_tgt_len: int = 128,
    ):
        self.model      = model.eval()
        self.tokenizer  = tokenizer
        self.device     = device
        self.temperature = temperature
        self.top_p      = top_p
        self.beam_width = beam_width
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_dir: str,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[dict] = None,
    ) -> "HappyBotInference":
        """Convenience factory: load tokenizer + model in one call."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer  = HappyBotTokenizer(tokenizer_dir)
        vocab_size = tokenizer.vocab_size

        defaults = dict(
            vocab_size=vocab_size, d_model=256, num_heads=2,
            num_encoder_layers=4, num_decoder_layers=4, d_ff=1024,
            dropout=0.0,   # disable dropout at inference
            num_emotion_classes=8, num_strategy_classes=8,
            pad_token_id=tokenizer.pad_id,
        )
        if model_kwargs:
            defaults.update(model_kwargs)

        model = HappyBot(**defaults).to(device)
        load_checkpoint(checkpoint_path, model, device=device)
        model.eval()

        return cls(model, tokenizer, device)

    # ─────────────────────────────────────────────────────────────────────
    # Inference pipeline
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        user_input: str,
        emotion_context: Optional[str] = None,  # e.g., "anxiety"
        intensity: Optional[int] = None,        # 1-5
        force_strategy: Optional[str] = None,   # override predicted strategy
    ) -> dict:
        """
        Full inference pipeline.

        Args:
            user_input:      Raw user text (conversation context).
            emotion_context: Optional emotion label to prepend (override).
            intensity:       Optional intensity token (override).
            force_strategy:  If provided, skip Strategy Head and use this strategy.

        Returns:
            {
                "strategy":         str (e.g., "reflection"),
                "strategy_id":      int,
                "emotion_logits":   list of floats,
                "response":         str,
                "encoder_attn":     list (attention weights for viz),
                "cross_attn":       list (cross-attention weights for viz),
            }
        """
        # ── Build encoder input ───────────────────────────────────────────
        # Prepend optional metadata tokens
        prefix = ""
        if emotion_context:
            prefix += f"[seeker_emotion_{emotion_context.lower()}]"
        if intensity:
            prefix += f"[intensity_{max(1, min(5, intensity))}]"

        full_input = f"{prefix} {user_input}".strip()

        # Prepend [CLS] so encoder can extract classification vector
        enc_ids = [self.tokenizer.cls_id] + self.tokenizer.encode(
            full_input, max_length=self.max_src_len - 1
        )
        enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=self.device)

        # ── Encode ────────────────────────────────────────────────────────
        H_enc, emotion_logits, strategy_logits, enc_pad_mask = self.model.encode(enc_tensor)

        # ── Strategy prediction ───────────────────────────────────────────
        if force_strategy:
            from tokenizer import STRATEGY_TO_ID
            strategy_id  = STRATEGY_TO_ID.get(force_strategy, 0)
            strategy_key = force_strategy
        else:
            strategy_id  = strategy_logits.argmax(dim=-1).item()
            strategy_key = ID_TO_STRATEGY.get(strategy_id, "other")

        strategy_token_id = self.tokenizer.strategy_token_id(strategy_key)

        # ── Seed decoder: [BOS][strategy_token] ──────────────────────────
        seed_ids = [self.tokenizer.bos_id, strategy_token_id]

        # ── Decode ────────────────────────────────────────────────────────
        use_beam = strategy_key in self.BEAM_STRATEGIES

        if use_beam:
            # Beam search for action-oriented strategies
            output_ids = beam_search(
                model=self.model,
                H_enc=H_enc,
                enc_pad_mask=enc_pad_mask,
                seed_ids=seed_ids,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                max_length=self.max_tgt_len,
                beam_width=self.beam_width,
                device=self.device,
            )
        else:
            # Nucleus sampling for empathetic/reflective strategies
            output_ids = seed_ids[:]
            cross_attn_all = []

            for _ in range(self.max_tgt_len):
                dec_ids = torch.tensor([output_ids], dtype=torch.long, device=self.device)
                logits_last, cross_attn = self.model.decode_step(
                    dec_ids, H_enc, enc_pad_mask
                )
                cross_attn_all = cross_attn   # keep last step's weights

                next_tok = top_p_sampling(
                    logits_last, self.temperature, self.top_p
                )
                output_ids.append(next_tok)

                if next_tok == self.tokenizer.eos_id:
                    break

        # ── Decode token IDs to text ──────────────────────────────────────
        # Skip seed tokens ([BOS] + strategy) when decoding final text
        response_ids = output_ids[len(seed_ids):]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        return {
            "strategy":        strategy_key,
            "strategy_id":     strategy_id,
            "emotion_logits":  emotion_logits.cpu().tolist(),
            "response":        response_text.strip(),
        }

    @torch.no_grad()
    def generate_with_attention(
        self,
        user_input: str,
        **kwargs,
    ) -> dict:
        """
        Same as generate() but also returns attention weights for visualization.
        Uses nucleus sampling only (beam search doesn't return per-step weights).
        """
        result = self.generate(user_input, **kwargs)

        # Run a second forward pass to collect attention weights on full sequence
        prefix = ""
        if kwargs.get("emotion_context"):
            prefix = f"[seeker_emotion_{kwargs['emotion_context']}]"
        full_input = f"{prefix} {user_input}".strip()
        enc_ids = [self.tokenizer.cls_id] + self.tokenizer.encode(full_input)
        enc_tensor = torch.tensor([enc_ids], dtype=torch.long, device=self.device)

        out = self.model(
            encoder_ids=enc_tensor,
            decoder_input=enc_tensor[:, :1],  # dummy 1-token decoder input
        )
        result["encoder_attn"] = [a.cpu() for a in out["encoder_attn"]]
        result["cross_attn"]   = [a.cpu() for a in out["cross_attn"]]
        result["encoder_tokens"] = [
            self.tokenizer.decode([i], skip_special_tokens=False)
            for i in enc_ids
        ]
        return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Happy-Bot Inference")
    p.add_argument("--checkpoint",      required=True)
    p.add_argument("--tokenizer_dir",   default="data/tokenizer")
    p.add_argument("--input",           default=None,
                   help="Single input string (interactive if omitted)")
    p.add_argument("--emotion",         default=None,
                   help="Override emotion context (e.g., anxiety)")
    p.add_argument("--intensity",       type=int, default=None)
    p.add_argument("--force_strategy",  default=None)
    p.add_argument("--temperature",     type=float, default=0.85)
    p.add_argument("--top_p",           type=float, default=0.90)
    p.add_argument("--beam_width",      type=int,   default=4)
    p.add_argument("--max_tgt_len",     type=int,   default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    engine = HappyBotInference.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        device=device,
    )
    engine.temperature = args.temperature
    engine.top_p       = args.top_p
    engine.beam_width  = args.beam_width
    engine.max_tgt_len = args.max_tgt_len

    def run_once(user_input: str):
        result = engine.generate(
            user_input,
            emotion_context=args.emotion,
            intensity=args.intensity,
            force_strategy=args.force_strategy,
        )
        print(f"\n[Strategy Predicted]: {result['strategy'].upper()}")
        print(f"[Response]: {result['response']}\n")

    if args.input:
        run_once(args.input)
    else:
        print("Happy-Bot interactive mode. Type 'exit' to quit.\n")
        while True:
            try:
                user_in = input("You: ").strip()
                if user_in.lower() in ("exit", "quit", "q"):
                    break
                if not user_in:
                    continue
                run_once(user_in)
            except (KeyboardInterrupt, EOFError):
                break
        print("Goodbye.")


if __name__ == "__main__":
    main()
