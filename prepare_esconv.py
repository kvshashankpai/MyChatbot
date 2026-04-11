"""
prepare_esconv.py — Normalise ESConv.json and produce Phase 2 training JSONL

Usage:
    python prepare_esconv.py \
        --input   data/raw/ESConv.json \
        --out_dir data/processed

Output files:
    data/processed/esconv_train.jsonl
    data/processed/esconv_val.jsonl
    data/processed/esconv_test.jsonl
    data/processed/strategy_counts.json

What this script does
─────────────────────
The raw ESConv.json schema does not exactly match what dataset.extract_esconv_qa_pairs()
expects. This script normalises the raw data first, then delegates all QA-pair
extraction, token-budget management, and label encoding to the existing
dataset.py function so there is a single source of truth.

Normalisation applied to each conversation
───────────────────────────────────────────
1. Role names:   "speaker"  → "seeker"
                 "listener" → "supporter"
2. Strategy:     moved from turn["annotation"]["strategy"] → turn["strategy"]
                 Raw variants mapped to ESCONV_STRATEGY_MAP keys:
                   "Other" / "Others"             → "Others"
                   "Questions" / "Question"        → "Questions"
                   "Approval and Reassurance"      → "Affirmation and Reassurance"
                   "Restatement"                   → "Restatement or Paraphrasing"
                   "Reflection of feelings"        → "Reflection of Feelings"
                   "Direct Guidance"               → "Providing Suggestions"
3. Intensity:    field absent in this release → default 3
4. Turns with no strategy annotation get "Others"

Emotion coverage
────────────────
ESCONV_EMOTION_TO_ID covers: anxiety, sadness, anger, fear, disgust, joy, surprise, neutral
ESConv emotions not in that map: depression, shame, nervousness
These are mapped to neutral (id=7) by ESCONV_EMOTION_TO_ID.get(x, 7) inside dataset.py.

Split
─────
Conversations are split 70/15/15 (train/val/test) at conversation level.
No conversation leaks across splits.
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset import extract_esconv_qa_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Raw strategy → ESCONV_STRATEGY_MAP key normalisation
# ─────────────────────────────────────────────────────────────────────────────

RAW_STRATEGY_NORM = {
    "other":                        "Others",
    "others":                       "Others",
    "questions":                    "Questions",
    "question":                     "Questions",
    "restatement or paraphrasing":  "Restatement or Paraphrasing",
    "restatement":                  "Restatement or Paraphrasing",
    "reflection of feelings":       "Reflection of Feelings",
    "affirmation and reassurance":  "Affirmation and Reassurance",
    "approval and reassurance":     "Affirmation and Reassurance",
    "providing suggestions":        "Providing Suggestions",
    "direct guidance":              "Providing Suggestions",
    "information":                  "Information",
    "self-disclosure":              "Self-disclosure",
}


def normalise_strategy(raw: str) -> str:
    return RAW_STRATEGY_NORM.get(raw.strip().lower(), "Others")


def normalise_conversation(conv: dict) -> dict:
    """
    Convert one raw ESConv conversation dict into the schema that
    extract_esconv_qa_pairs() expects.
    """
    norm_turns = []
    for turn in conv.get("dialog", []):
        raw_speaker  = turn.get("speaker", "seeker")
        role         = "seeker" if raw_speaker == "speaker" else "supporter"
        content      = turn.get("content", "").strip()
        raw_strategy = turn.get("annotation", {}).get("strategy") or "Others"
        strategy     = normalise_strategy(raw_strategy)

        norm_turns.append({
            "role":     role,
            "content":  content,
            "strategy": strategy,
        })

    return {
        "emotion_type":      conv.get("emotion_type", "neutral").lower().strip(),
        "situation":         conv.get("situation", "").strip(),
        "emotion_intensity": int(conv.get("emotion_intensity", 3)),
        "dialog":            norm_turns,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess ESConv.json for Phase 2")
    parser.add_argument("--input",     default="data/raw/ESConv.json")
    parser.add_argument("--out_dir",   default="data/processed")
    parser.add_argument("--val_frac",  type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load and normalise ─────────────────────────────────────────────────
    with open(args.input, encoding="utf-8") as f:
        raw_conversations = json.load(f)

    print(f"Loaded {len(raw_conversations)} conversations from {args.input}")
    normalised = [normalise_conversation(c) for c in raw_conversations]

    # ── Train / val / test split at conversation level ─────────────────────
    indices = list(range(len(normalised)))
    random.shuffle(indices)
    n_test  = max(1, int(len(indices) * args.test_frac))
    n_val   = max(1, int(len(indices) * args.val_frac))
    test_idx = set(indices[:n_test])
    val_idx  = set(indices[n_test:n_test + n_val])

    train_convs = [normalised[i] for i in indices[n_test + n_val:]]
    val_convs   = [normalised[i] for i in val_idx]
    test_convs  = [normalised[i] for i in test_idx]

    print(f"Split: {len(train_convs)} train | {len(val_convs)} val | {len(test_convs)} test conversations")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Write normalised JSON to temp files, then extract QA pairs ─────────
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        json.dump(train_convs, tf)
        train_tmp = tf.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        json.dump(val_convs, tf)
        val_tmp = tf.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
        json.dump(test_convs, tf)
        test_tmp = tf.name

    train_out = str(out / "esconv_train.jsonl")
    val_out   = str(out / "esconv_val.jsonl")
    test_out  = str(out / "esconv_test.jsonl")

    # extract_esconv_qa_pairs handles: sliding window, metadata tokens,
    # strategy token prepended to target, emotion/strategy integer IDs.
    print("Extracting train QA pairs ...")
    strategy_counter = extract_esconv_qa_pairs(
        esconv_json_path=train_tmp,
        output_path=train_out,
        window_size=3,
    )

    print("Extracting val QA pairs ...")
    extract_esconv_qa_pairs(
        esconv_json_path=val_tmp,
        output_path=val_out,
        window_size=3,
    )

    print("Extracting test QA pairs ...")
    extract_esconv_qa_pairs(
        esconv_json_path=test_tmp,
        output_path=test_out,
        window_size=3,
    )

    os.unlink(train_tmp)
    os.unlink(val_tmp)
    os.unlink(test_tmp)

    # ── Write strategy counts for class-weighted loss ──────────────────────
    counts_path = str(out / "strategy_counts.json")
    with open(counts_path, "w") as f:
        json.dump(dict(strategy_counter), f, indent=2)
    print(f"Strategy counts -> {counts_path}")
    print(" ", dict(strategy_counter))

    # ── Summary ────────────────────────────────────────────────────────────
    with open(train_out) as f:
        n_train = sum(1 for _ in f)
    with open(val_out) as f:
        n_val_pairs = sum(1 for _ in f)
    with open(test_out) as f:
        n_test_pairs = sum(1 for _ in f)
    print(f"\nDone. Train pairs: {n_train}, Val pairs: {n_val_pairs}, Test pairs: {n_test_pairs}")
    print(f"Files written to {out}/")


if __name__ == "__main__":
    main()