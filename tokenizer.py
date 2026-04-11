"""
tokenizer.py — Custom BPE tokenizer training for HappyBot.

Trains a Byte-Pair Encoding tokenizer on the project corpus with:
  - Vocab size ~10,000
  - All special/control tokens pre-reserved BEFORE BPE merges are learned
    (critical: prevents merges from absorbing special token substrings)
  - Special tokens cover: structural tokens, 32 emotion labels,
    8 strategy labels, and intensity markers

Usage:
  python tokenizer.py --corpus data/processed/corpus.txt --output_dir data/tokenizer
"""

import argparse
import os
from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders, models


# ── Special Token Definitions ─────────────────────────────────────────────────

STRUCTURAL_TOKENS = [
    "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[CLS]",
    "[SEEKER]:", "[SUPPORTER]:", "[SITUATION]:",
]

EMOTION_TOKENS = [
    "[seeker_emotion_SENTIMENTAL]", "[seeker_emotion_AFRAID]", "[seeker_emotion_PROUD]",
    "[seeker_emotion_FAITHFUL]", "[seeker_emotion_TERRIFIED]", "[seeker_emotion_JOYFUL]",
    "[seeker_emotion_ANGRY]", "[seeker_emotion_SAD]", "[seeker_emotion_JEALOUS]",
    "[seeker_emotion_GRATEFUL]", "[seeker_emotion_PREPARED]", "[seeker_emotion_EMBARRASSED]",
    "[seeker_emotion_EXCITED]", "[seeker_emotion_ANNOYED]", "[seeker_emotion_LONELY]",
    "[seeker_emotion_ASHAMED]", "[seeker_emotion_SURPRISED]", "[seeker_emotion_DISGUSTED]",
    "[seeker_emotion_ANTICIPATING]", "[seeker_emotion_CONFIDENT]", "[seeker_emotion_FURIOUS]",
    "[seeker_emotion_DEVASTATED]", "[seeker_emotion_HOPEFUL]", "[seeker_emotion_ANXIOUS]",
    "[seeker_emotion_TRUSTING]", "[seeker_emotion_CONTENT]", "[seeker_emotion_IMPRESSED]",
    "[seeker_emotion_APPREHENSIVE]", "[seeker_emotion_CARING]", "[seeker_emotion_GUILTY]",
    "[seeker_emotion_CURIOUS]", "[seeker_emotion_NEUTRAL]",
    # ESConv emotion tokens
    "[seeker_emotion_ANXIETY]", "[seeker_emotion_DEPRESSION]", "[seeker_emotion_SADNESS]",
    "[seeker_emotion_ANGER]", "[seeker_emotion_FEAR]", "[seeker_emotion_DISGUST]",
]

STRATEGY_TOKENS = [
    "[strategy_QUESTION]",
    "[strategy_RESTATEMENT]",
    "[strategy_AFFIRMATION_AND_REASSURANCE]",
    "[strategy_SUGGESTION]",
    "[strategy_INFORMATION]",
    "[strategy_SELF_DISCLOSURE]",
    "[strategy_OTHERS_EXPERIENCES]",
    "[strategy_TRANSITION_TO_PROBLEM]",
]

INTENSITY_TOKENS = [f"[intensity_{i}]" for i in range(1, 6)]

ALL_SPECIAL_TOKENS = (
    STRUCTURAL_TOKENS + EMOTION_TOKENS + STRATEGY_TOKENS + INTENSITY_TOKENS
)


def train_tokenizer(corpus_path: str, output_dir: str, vocab_size: int = 10_000):
    """
    Train a BPE tokenizer on the project corpus.
    
    CRITICAL: Special tokens are added BEFORE training so that BPE merges
    never split them into subword pieces.
    """
    print(f"[Tokenizer] Training BPE on {corpus_path}")
    print(f"[Tokenizer] Target vocab size: {vocab_size}")
    print(f"[Tokenizer] Special tokens: {len(ALL_SPECIAL_TOKENS)}")

    # ── Build BPE tokenizer ──────────────────────────────────────────────
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Byte-level pre-tokenizer: handles whitespace correctly
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=ALL_SPECIAL_TOKENS,
        min_frequency=2,    # Only form merges for pairs seen ≥ 2 times
        show_progress=True,
    )

    # ── Train ───────────────────────────────────────────────────────────
    tokenizer.train([corpus_path], trainer)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"[Tokenizer] Trained. Actual vocab size: {actual_vocab}")

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(out_path)
    print(f"[Tokenizer] Saved → {out_path}")

    # Also save a human-readable vocab JSON for debugging
    vocab = tokenizer.get_vocab()
    import json
    vocab_readable_path = os.path.join(output_dir, "vocab_readable.json")
    with open(vocab_readable_path, "w", encoding="utf-8") as f:
        # Sort by id for readability
        json.dump({v: k for k, v in sorted(vocab.items(), key=lambda x: x[1])}, f, indent=2)
    print(f"[Tokenizer] Vocab readable → {vocab_readable_path}")

    # ── Verify round-trip ───────────────────────────────────────────────
    test_inputs = [
        "[CLS][seeker_emotion_SAD][intensity_3] [SEEKER]: I feel really lost and alone.",
        "[BOS][strategy_QUESTION] How long have you been feeling this way? [EOS]",
    ]
    print("\n[Tokenizer] Verification round-trip:")
    for test in test_inputs:
        enc = tokenizer.encode(test)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        ok = "✓" if all(t in enc.tokens for t in ["[CLS]", "[SEEKER]:"] if t in test) else "?"
        print(f"  {ok} Input : {test[:60]}...")
        print(f"    Tokens: {len(enc.ids)} | First 5: {enc.tokens[:5]}")

    return tokenizer


def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load a saved tokenizer from disk."""
    path = os.path.join(tokenizer_dir, "tokenizer.json")
    return Tokenizer.from_file(path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus",      default="data/processed/corpus.txt")
    p.add_argument("--output_dir",  default="data/tokenizer")
    p.add_argument("--vocab_size",  type=int, default=10_000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(args.corpus, args.output_dir, args.vocab_size)