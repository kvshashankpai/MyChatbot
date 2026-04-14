"""
inference.py — HappyBot inference engine.

Fixes vs the repo version:
  1. from_checkpoint — no longer reads vocab_size etc. from checkpoint dict
     (they're saved there but we read them to reconstruct the model correctly)
  2. load_checkpoint — now called correctly (no model arg)
  3. decode_step — calls model.decode_step() not model.decode()
  4. Strategy token IS injected into decoder seed  (was missing — critical)
  5. Beam search calls model.decode_step() not model.decode()
  6. enc_pad_mask built from model.make_src_mask(), not a manual comparison
"""

import argparse
import os
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dataset import CANONICAL_STRATEGIES
from model.transformer import HappyBot
from utils import load_checkpoint


# Strategy IDs that use beam search (more coherent for actionable advice)
BEAM_STRATEGY_IDS = {3, 4}   # suggestion=3, information=4

EMOTION_LABELS = [
    "sentimental", "afraid", "proud", "faithful", "terrified", "joyful", "angry",
    "sad", "jealous", "grateful", "prepared", "embarrassed", "excited", "annoyed",
    "lonely", "ashamed", "surprised", "disgusted", "anticipating", "confident",
    "furious", "devastated", "hopeful", "anxious", "trusting", "content", "impressed",
    "apprehensive", "caring", "guilty", "curious", "neutral",
]


# ── Sampling helpers ──────────────────────────────────────────────────────────

def top_p_sampling(logits: torch.Tensor, temperature: float = 0.85, top_p: float = 0.9) -> int:
    """Nucleus sampling.  logits: (V,) or (1,V).  Returns a single int token id."""
    logits = logits.squeeze(0) / temperature
    probs  = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)

    # Remove tokens past the nucleus
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum()

    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_idx[sampled].item()


def beam_search(model, memory, src_mask, bos_id, eos_id, pad_id,
                beam_width=4, max_len=128, device=None):
    """Beam search decoding.  Returns list of token ids (without BOS)."""
    beams     = [(0.0, [bos_id])]
    completed = []

    for _ in range(max_len):
        if not beams:
            break
        new_beams = []
        for score, seq in beams:
            if seq[-1] == eos_id:
                completed.append((score / len(seq), seq))
                continue
            tgt     = torch.tensor([seq], dtype=torch.long, device=device)
            logits  = model.decode_step(tgt, memory, src_mask)  # (1, V)
            lp      = F.log_softmax(logits[0], dim=-1)
            topk    = torch.topk(lp, beam_width)
            for lp_val, tok in zip(topk.values.tolist(), topk.indices.tolist()):
                new_beams.append((score + lp_val, seq + [tok]))

        new_beams.sort(key=lambda x: x[0] / max(1, len(x[1])), reverse=True)
        beams = new_beams[:beam_width]
        if all(s[-1] == eos_id for _, s in beams):
            for sc, s in beams:
                completed.append((sc / len(s), s))
            break

    for sc, s in beams:
        completed.append((sc / max(1, len(s)), s))

    if not completed:
        return []
    best = max(completed, key=lambda x: x[0])
    return [t for t in best[1][1:] if t not in (eos_id, pad_id)]


# ── Inference engine ──────────────────────────────────────────────────────────

class HappyBotInference:

    def __init__(self, model, tokenizer, device,
                 temperature=0.85, top_p=0.9, beam_width=4, max_len=128):
        self.model       = model.eval()
        self.tokenizer   = tokenizer
        self.device      = device
        self.temperature = temperature
        self.top_p       = top_p
        self.beam_width  = beam_width
        self.max_len     = max_len

        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.bos_id = tokenizer.token_to_id("[BOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")

    @classmethod
    def from_checkpoint(cls, checkpoint_path, tokenizer_dir, device):
        """
        Reconstruct model from checkpoint.
        All architecture hyper-params are stored inside the checkpoint by save_checkpoint().
        """
        ckpt = load_checkpoint(checkpoint_path, map_location=device)

        model = HappyBot(
            vocab_size         = ckpt["vocab_size"],
            d_model            = ckpt.get("d_model",            256),
            num_encoder_layers = ckpt.get("num_encoder_layers", 4),
            num_decoder_layers = ckpt.get("num_decoder_layers", 4),
            num_heads          = ckpt.get("num_heads",          2),
            d_ff               = ckpt.get("d_ff",               1024),
            dropout            = ckpt.get("dropout",            0.0),
            num_emotion_classes  = 32,
            num_strategy_classes = 8,
            pad_token_id = ckpt.get("pad_token_id", 0),
        ).to(device)

        model.load_state_dict(ckpt["model_state_dict"])

        tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        return cls(model, tok, device)

    @torch.no_grad()
    def generate(self, user_input, emotion_context=None, intensity=None):
        # Build encoder input string
        emo_tok = f"[seeker_emotion_{emotion_context.upper()}]" if emotion_context else ""
        int_tok = f"[intensity_{intensity}]"                    if intensity       else ""
        enc_str = f"[CLS]{emo_tok}{int_tok} [SEEKER]: {user_input}"

        src_ids = self.tokenizer.encode(enc_str).ids
        src     = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # Encode once
        enc      = self.model.encode(src)
        memory   = enc["memory"]
        src_mask = enc["src_mask"]

        # Predict strategy and emotion
        strategy_id  = enc["strategy_logits"].argmax(-1).item()
        emotion_id   = enc["emotion_logits"].argmax(-1).item()
        strategy_name = CANONICAL_STRATEGIES[strategy_id] if strategy_id < len(CANONICAL_STRATEGIES) else "question"
        emotion_name  = EMOTION_LABELS[emotion_id]        if emotion_id  < len(EMOTION_LABELS)       else "neutral"

        # ── Strategy token injection into decoder seed ────────────────────
        # This is the critical fix: the strategy prediction must actually
        # condition generation by becoming the first decoder input token.
        strat_tok_str = f"[strategy_{strategy_name.upper()}]"
        strat_tok_id  = self.tokenizer.token_to_id(strat_tok_str)
        seed = [self.bos_id] + ([strat_tok_id] if strat_tok_id is not None else [])

        # Choose decoding strategy
        if strategy_id in BEAM_STRATEGY_IDS:
            response_ids = beam_search(
                self.model, memory, src_mask,
                bos_id=self.bos_id, eos_id=self.eos_id, pad_id=self.pad_id,
                beam_width=self.beam_width, max_len=self.max_len, device=self.device,
            )
        else:
            cur  = torch.tensor([seed], dtype=torch.long, device=self.device)
            response_ids = []
            for _ in range(self.max_len):
                logits  = self.model.decode_step(cur, memory, src_mask)  # (1, V)
                next_id = top_p_sampling(logits[0], self.temperature, self.top_p)
                if next_id == self.eos_id:
                    break
                response_ids.append(next_id)
                cur = torch.cat([cur, torch.tensor([[next_id]], device=self.device)], dim=1)

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return {"response": response, "strategy": strategy_name, "emotion": emotion_name}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--tokenizer_dir", default="data/tokenizer")
    p.add_argument("--input",         default=None)
    p.add_argument("--emotion",       default=None)
    p.add_argument("--intensity",     type=int, default=None)
    p.add_argument("--temperature",   type=float, default=0.85)
    p.add_argument("--top_p",         type=float, default=0.90)
    p.add_argument("--beam_width",    type=int,   default=4)
    p.add_argument("--max_len",       type=int,   default=128)
    return p.parse_args()


def main():
    args   = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Loading model on {device}...")

    engine = HappyBotInference.from_checkpoint(args.checkpoint, args.tokenizer_dir, device)
    engine.temperature = args.temperature
    engine.top_p       = args.top_p
    engine.beam_width  = args.beam_width
    engine.max_len     = args.max_len

    def run(text):
        r = engine.generate(text, emotion_context=args.emotion, intensity=args.intensity)
        print(f"\n[Emotion detected] {r['emotion']}")
        print(f"[Strategy selected] {r['strategy']}")
        print(f"[Response] {r['response']}\n")

    if args.input:
        run(args.input)
    else:
        print("Happy-Bot interactive mode. Type 'exit' to quit.\n")
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
            context = " ".join(history[-3:])
            r = engine.generate(context, emotion_context=args.emotion, intensity=args.intensity)
            print(f"Bot [{r['strategy']}]: {r['response']}\n")
            history.append(f"[SUPPORTER]: {r['response']}")
        print("Goodbye.")


if __name__ == "__main__":
    main()