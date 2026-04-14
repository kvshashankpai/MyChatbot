"""
inference.py — HappyBot inference engine.

FIXES vs original:
  1. Default temperature lowered from 0.85 → 0.75 and top_p from 0.90 → 0.85.
     Higher temperature caused the small model to explore too much probability
     mass, producing incoherent word salad. Lower temperature sharpens the
     distribution around likely next tokens.

  2. Repetition penalty added to nucleus sampling (default 1.3).
     Without this, the model loops on short repeated phrases — the classic
     mode collapse symptom visible in the original broken output.

  3. Beam search now injects the strategy token into the seed, matching the
     training decoder_input format: [BOS][strategy_token] ...
     The original beam search started from [BOS] only, so strategy-conditioned
     beams (suggestion, information) were never properly grounded.

  4. Beam search applies length normalization penalty (alpha=0.6) to prevent
     the model preferring very short beams with slightly better per-token logprob.

  5. from_checkpoint() dropout set to 0.0 at inference time (was inheriting
     whatever was saved in the checkpoint).

Usage:
  python inference.py --checkpoint checkpoints/phase2/best_model.pt

  Interactive mode (default):
    python inference.py --checkpoint checkpoints/phase2/best_model.pt

  Single-turn:
    python inference.py --checkpoint checkpoints/phase2/best_model.pt \\
        --input "I've been feeling really anxious lately about my job"

  With known emotion/intensity:
    python inference.py --checkpoint checkpoints/phase2/best_model.pt \\
        --input "I can't stop worrying" --emotion anxiety --intensity 4
"""

import argparse
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dataset import CANONICAL_STRATEGIES
from model.transformer import HappyBot
from utils import load_checkpoint


# Strategy IDs that use beam search (coherent for actionable advice)
BEAM_STRATEGY_IDS = {3, 4}   # suggestion=3, information=4

EMOTION_LABELS = [
    "sentimental", "afraid", "proud", "faithful", "terrified", "joyful", "angry",
    "sad", "jealous", "grateful", "prepared", "embarrassed", "excited", "annoyed",
    "lonely", "ashamed", "surprised", "disgusted", "anticipating", "confident",
    "furious", "devastated", "hopeful", "anxious", "trusting", "content", "impressed",
    "apprehensive", "caring", "guilty", "curious", "neutral",
]


# ── Sampling helpers ──────────────────────────────────────────────────────────

def top_p_sampling(
    logits: torch.Tensor,
    temperature: float = 0.75,
    top_p: float = 0.85,
    repetition_penalty: float = 1.3,
    generated_ids: list = None,
) -> int:
    """
    Nucleus (top-p) sampling with repetition penalty.

    logits         : (V,) or (1, V) — raw logits for the next token
    repetition_penalty : values > 1 discourage tokens already generated.
                        1.0 = disabled. 1.2–1.5 is a good range.
    generated_ids  : list of already-generated token ids for penalty application.
    """
    logits = logits.squeeze(0).clone()

    # FIX: Repetition penalty — divide logits of already-seen tokens
    if repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= repetition_penalty
            else:
                logits[token_id] *= repetition_penalty

    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # Remove tokens past the nucleus
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum().clamp(min=1e-8)

    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[sampled].item()


def beam_search(
    model,
    memory,
    src_mask,
    bos_id,
    eos_id,
    pad_id,
    strategy_token_id=None,
    beam_width=5,
    max_len=128,
    length_penalty_alpha=0.6,
    device=None,
):
    """
    Beam search decoding with:
      - FIX: strategy token seed injection (matching training format)
      - FIX: length normalisation penalty (alpha=0.6, Wu et al. 2016)

    Returns list of token ids (without BOS/strategy seed tokens).
    """
    # FIX: seed includes strategy token when available
    seed = [bos_id]
    if strategy_token_id is not None:
        seed.append(strategy_token_id)

    beams     = [(0.0, seed)]
    completed = []

    for _ in range(max_len):
        if not beams:
            break
        new_beams = []
        for score, seq in beams:
            if seq[-1] == eos_id:
                completed.append((score, seq))
                continue
            tgt    = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model.decode_step(tgt, memory, src_mask)   # (1, V)
            lp     = F.log_softmax(logits[0], dim=-1)
            topk   = torch.topk(lp, beam_width)
            for lp_val, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                new_beams.append((score + lp_val, seq + [tok]))

        # FIX: length normalisation — avoids bias toward short beams
        def norm_score(sc, seq):
            return sc / (len(seq) ** length_penalty_alpha)

        new_beams.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
        beams = new_beams[:beam_width]
        if all(s[-1] == eos_id for _, s in beams):
            for sc, s in beams:
                completed.append((sc, s))
            break

    for sc, s in beams:
        completed.append((sc, s))

    if not completed:
        return []

    # Pick best by length-normalised score
    best = max(completed, key=lambda x: norm_score(x[0], x[1]))
    # Strip seed tokens (BOS + optional strategy) and EOS/PAD
    seed_len    = len(seed)
    output_ids  = best[1][seed_len:]
    return [t for t in output_ids if t not in (eos_id, pad_id)]


# ── Inference engine ──────────────────────────────────────────────────────────

class HappyBotInference:

    def __init__(
        self,
        model,
        tokenizer,
        device,
        temperature=0.75,        # FIX: was 0.85
        top_p=0.85,              # FIX: was 0.90
        repetition_penalty=1.3,  # FIX: new — suppresses repetitive loops
        beam_width=5,            # FIX: was 4
        max_len=128,
    ):
        self.model              = model.eval()
        self.tokenizer          = tokenizer
        self.device             = device
        self.temperature        = temperature
        self.top_p              = top_p
        self.repetition_penalty = repetition_penalty
        self.beam_width         = beam_width
        self.max_len            = max_len

        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")

        # Pre-build strategy token id map
        self.strat_token_ids = {}
        for i, s in enumerate(CANONICAL_STRATEGIES):
            self.strat_token_ids[i] = tokenizer.token_to_id(f"[strategy_{s.upper()}]")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer_dir, device, **kwargs):
        """Reconstruct model from checkpoint."""
        ckpt = load_checkpoint(checkpoint_path, map_location=device)

        model = HappyBot(
            vocab_size          = ckpt["vocab_size"],
            d_model             = ckpt.get("d_model",            256),
            num_encoder_layers  = ckpt.get("num_encoder_layers", 4),
            num_decoder_layers  = ckpt.get("num_decoder_layers", 4),
            num_heads           = ckpt.get("num_heads",          2),
            d_ff                = ckpt.get("d_ff",               1024),
            dropout             = 0.0,    # FIX: always 0 at inference time
            num_emotion_classes  = 32,
            num_strategy_classes = 8,
            pad_token_id = ckpt.get("pad_token_id", 0),
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        return cls(model, tok, device, **kwargs)

    @torch.no_grad()
    def generate(self, user_input, emotion_context=None, intensity=None):
        # Build encoder input
        emo_tok = f"[seeker_emotion_{emotion_context.upper()}]" if emotion_context else ""
        int_tok = f"[intensity_{intensity}]"                    if intensity       else ""
        enc_str = f"[CLS]{emo_tok}{int_tok} [SEEKER]: {user_input}"

        src_ids = self.tokenizer.encode(enc_str).ids
        src     = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # Encode once
        enc      = self.model.encode(src)
        memory   = enc["memory"]
        src_mask = enc["src_mask"]

        # Predict strategy and emotion from encoder
        strategy_id   = enc["strategy_logits"].argmax(-1).item()
        emotion_id    = enc["emotion_logits"].argmax(-1).item()
        strategy_name = CANONICAL_STRATEGIES[strategy_id] if strategy_id < len(CANONICAL_STRATEGIES) else "question"
        emotion_name  = EMOTION_LABELS[emotion_id]        if emotion_id  < len(EMOTION_LABELS)       else "neutral"

        # Get strategy token id for decoder seed
        strat_tok_id = self.strat_token_ids.get(strategy_id)

        # Choose decoding strategy
        if strategy_id in BEAM_STRATEGY_IDS:
            # FIX: beam search now uses strategy seed token
            response_ids = beam_search(
                self.model, memory, src_mask,
                bos_id=self.bos_id, eos_id=self.eos_id, pad_id=self.pad_id,
                strategy_token_id=strat_tok_id,
                beam_width=self.beam_width,
                max_len=self.max_len,
                device=self.device,
            )
        else:
            # Nucleus sampling with seed: [BOS, strategy_token]
            seed = [self.bos_id]
            if strat_tok_id is not None:
                seed.append(strat_tok_id)
            cur          = torch.tensor([seed], dtype=torch.long, device=self.device)
            response_ids = []

            for _ in range(self.max_len):
                logits  = self.model.decode_step(cur, memory, src_mask)   # (1, V)
                # FIX: pass generated ids for repetition penalty
                next_id = top_p_sampling(
                    logits[0],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    generated_ids=response_ids,
                )
                if next_id == self.eos_id:
                    break
                response_ids.append(next_id)
                cur = torch.cat([cur, torch.tensor([[next_id]], device=self.device)], dim=1)

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return {"response": response, "strategy": strategy_name, "emotion": emotion_name}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--tokenizer_dir",       default="data/tokenizer")
    p.add_argument("--input",               default=None)
    p.add_argument("--emotion",             default=None)
    p.add_argument("--intensity",           type=int, default=None)
    p.add_argument("--temperature",         type=float, default=0.75)   # FIX
    p.add_argument("--top_p",               type=float, default=0.85)   # FIX
    p.add_argument("--repetition_penalty",  type=float, default=1.3)    # FIX: new
    p.add_argument("--beam_width",          type=int,   default=5)      # FIX: was 4
    p.add_argument("--max_len",             type=int,   default=128)
    return p.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Loading model on {device}...")

    engine = HappyBotInference.from_checkpoint(
        args.checkpoint,
        args.tokenizer_dir,
        device,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        beam_width=args.beam_width,
        max_len=args.max_len,
    )

    def run(text):
        r = engine.generate(text, emotion_context=args.emotion, intensity=args.intensity)
        print(f"\n[Emotion detected] {r['emotion']}")
        print(f"[Strategy selected] {r['strategy']}")
        print(f"[Response] {r['response']}\n")

    if args.input:
        run(args.input)
    else:
        print("HappyBot interactive mode. Type 'exit' to quit.\n")
        history = []
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user.lower() in ("exit", "quit", "q"):
                break
            if not user:
                continue
            history.append(f"[SEEKER]: {user}")
            # Use last 3 turns as context
            context = " ".join(history[-3:])
            r = engine.generate(context, emotion_context=args.emotion, intensity=args.intensity)
            print(f"Bot [{r['strategy']}]: {r['response']}\n")
            history.append(f"[SUPPORTER]: {r['response']}")
        print("Goodbye.")


if __name__ == "__main__":
    main()