"""
scripts/prepare_data.py — Master data preparation pipeline.

Runs all data preparation steps in sequence:
  1. Process EmpatheticDialogues CSV → JSONL splits
  2. Process ESConv → JSONL splits (via prepare_esconv.py)
  3. Build corpus.txt from all text
  4. Train BPE tokenizer on corpus.txt

Usage:
  python scripts/prepare_data.py
  python scripts/prepare_data.py --empathetic_csv data/raw/empathetic_dialogues/emotion-emotion_69k.csv
  python scripts/prepare_data.py --skip_tokenizer   # re-run data steps only
  python scripts/prepare_data.py --skip_empathetic  # re-run ESConv + tokenizer only
"""

import argparse
import csv
import json
import os
import sys
import random
from collections import defaultdict

# Add parent directory to path so we can import from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prepare_esconv import prepare_esconv
from tokenizer import train_tokenizer


# ── EmpatheticDialogues emotion label → integer id ────────────────────────────

ED_EMOTION_LABELS = [
    "sentimental", "afraid", "proud", "faithful", "terrified", "joyful", "angry",
    "sad", "jealous", "grateful", "prepared", "embarrassed", "excited", "annoyed",
    "lonely", "ashamed", "surprised", "disgusted", "anticipating", "confident",
    "furious", "devastated", "hopeful", "anxious", "trusting", "content", "impressed",
    "apprehensive", "caring", "guilty", "curious", "neutral",
]
ED_EMOTION_TO_ID = {e: i for i, e in enumerate(ED_EMOTION_LABELS)}


def process_empathetic_dialogues(csv_path: str, out_dir: str):
    """
    Process EmpatheticDialogues CSV into JSONL format.

    CSV columns: conv_id, utterance_idx, context, utterance, selfeval, tags
    Each conv_id has multiple rows (back-and-forth utterances).
    The 'context' column is the emotion label for the conversation.
    utterance_idx=1 is the seeker's initial message; idx=2 is supporter's first reply, etc.

    We extract (context_text, response) pairs where:
      context_text = [emotion_LABEL] seeker_utterance
      response = supporter_utterance
    And filter out exchanges shorter than 5 tokens.
    """
    print(f"[ED] Processing: {csv_path}")

    # Group by conv_id
    conversations = defaultdict(list)
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row.get("conv_id", "").strip()
            if not conv_id:
                continue
            idx = int(row.get("utterance_idx", 0) or 0)
            utterance = row.get("utterance", "").strip()
            emotion = row.get("context", "neutral").strip().lower()
            conversations[conv_id].append({
                "idx": idx,
                "utterance": utterance,
                "emotion": emotion,
            })

    print(f"[ED] Total conversations: {len(conversations)}")

    # Extract (input, target) pairs
    pairs = []
    for conv_id, turns in conversations.items():
        turns.sort(key=lambda t: t["idx"])
        emotion = turns[0]["emotion"] if turns else "neutral"
        emotion_label = ED_EMOTION_TO_ID.get(emotion, 31)  # default: neutral

        # Pair consecutive turns: seeker(i) → supporter(i+1)
        for i in range(0, len(turns) - 1, 2):
            seeker_utt   = turns[i]["utterance"]
            if i + 1 < len(turns):
                supporter_utt = turns[i + 1]["utterance"]
            else:
                continue

            # Filter very short exchanges
            if len(seeker_utt.split()) < 3 or len(supporter_utt.split()) < 3:
                continue

            input_text = f"[CLS][seeker_emotion_{emotion.upper()}] [SEEKER]: {seeker_utt}"
            pairs.append({
                "input":          input_text,
                "target":         supporter_utt,
                "emotion_label":  emotion_label,
                "strategy_label": -1,   # No strategy annotations in ED
            })

    print(f"[ED] Extracted {len(pairs)} training pairs")

    # Split: 80/10/10
    random.shuffle(pairs)
    n = len(pairs)
    n_val = n // 10
    n_test = n // 10
    splits = {
        "train": pairs[:n - n_val - n_test],
        "val":   pairs[n - n_val - n_test:n - n_test],
        "test":  pairs[n - n_test:],
    }

    os.makedirs(out_dir, exist_ok=True)
    for split_name, split_pairs in splits.items():
        path = os.path.join(out_dir, f"empathetic_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for pair in split_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"[ED] Wrote {len(split_pairs)} pairs → {path}")

    # Also write full (for corpus building)
    full_path = os.path.join(out_dir, "empathetic_full.jsonl")
    with open(full_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"[ED] Full set → {full_path}")

    return pairs


def build_corpus(processed_dir: str, corpus_path: str):
    """
    Concatenate all input and target text from all JSONL files into a single
    corpus.txt for BPE tokenizer training.
    """
    print(f"\n[Corpus] Building corpus at: {corpus_path}")
    files = [f for f in os.listdir(processed_dir) if f.endswith(".jsonl")]
    lines_written = 0

    with open(corpus_path, "w", encoding="utf-8") as out:
        for fname in sorted(files):
            fpath = os.path.join(processed_dir, fname)
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        # Write input and target as separate lines (cleaned)
                        inp = obj.get("input", "").strip()
                        tgt = obj.get("target", "").strip()
                        if inp:
                            out.write(inp + "\n")
                            lines_written += 1
                        if tgt:
                            out.write(tgt + "\n")
                            lines_written += 1
                    except json.JSONDecodeError:
                        continue

    print(f"[Corpus] Wrote {lines_written} lines to {corpus_path}")


def verify_outputs(processed_dir: str, tokenizer_dir: str) -> bool:
    required = [
        os.path.join(processed_dir, "empathetic_train.jsonl"),
        os.path.join(processed_dir, "empathetic_val.jsonl"),
        os.path.join(processed_dir, "empathetic_test.jsonl"),
        os.path.join(processed_dir, "esconv_train.jsonl"),
        os.path.join(processed_dir, "esconv_val.jsonl"),
        os.path.join(processed_dir, "esconv_test.jsonl"),
        os.path.join(processed_dir, "strategy_counts.json"),
        os.path.join(processed_dir, "corpus.txt"),
        os.path.join(tokenizer_dir, "tokenizer.json"),
    ]
    print("\n[Verify] Checking output files:")
    all_ok = True
    for path in required:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = "✓" if exists and size > 0 else "✗ MISSING"
        print(f"  {status}  {path}  ({size:,} bytes)")
        if not (exists and size > 0):
            all_ok = False
    return all_ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--empathetic_csv",  default="data/raw/empathetic_dialogues/emotion-emotion_69k.csv")
    p.add_argument("--esconv_path",     default="data/raw/ESConv.json")
    p.add_argument("--tokenizer_dir",   default="data/tokenizer")
    p.add_argument("--processed_dir",   default="data/processed")
    p.add_argument("--vocab_size",      type=int, default=10_000)
    p.add_argument("--skip_empathetic", action="store_true")
    p.add_argument("--skip_esconv",     action="store_true")
    p.add_argument("--skip_tokenizer",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.tokenizer_dir, exist_ok=True)

    # ── Step 1: EmpatheticDialogues ──────────────────────────────────────
    if not args.skip_empathetic:
        if not os.path.exists(args.empathetic_csv):
            print(f"[ERROR] EmpatheticDialogues CSV not found at: {args.empathetic_csv}")
            print("  Download from Kaggle and place at the path above.")
            sys.exit(1)
        process_empathetic_dialogues(args.empathetic_csv, args.processed_dir)
    else:
        print("[Skip] EmpatheticDialogues processing")

    # ── Step 2: ESConv ───────────────────────────────────────────────────
    if not args.skip_esconv:
        if not os.path.exists(args.esconv_path):
            print(f"[ERROR] ESConv.json not found at: {args.esconv_path}")
            print("  Clone from github.com/thu-coai/Emotional-Support-Conversation")
            sys.exit(1)
        prepare_esconv(args.esconv_path, args.processed_dir)
    else:
        print("[Skip] ESConv processing")

    # ── Step 3: Build corpus ─────────────────────────────────────────────
    corpus_path = os.path.join(args.processed_dir, "corpus.txt")
    if not args.skip_tokenizer:
        build_corpus(args.processed_dir, corpus_path)

    # ── Step 4: Train tokenizer ──────────────────────────────────────────
    if not args.skip_tokenizer:
        if not os.path.exists(corpus_path):
            print(f"[ERROR] Corpus not found at {corpus_path}. Run without --skip_tokenizer.")
            sys.exit(1)
        train_tokenizer(corpus_path, args.tokenizer_dir, args.vocab_size)

    # ── Verify all outputs ───────────────────────────────────────────────
    ok = verify_outputs(args.processed_dir, args.tokenizer_dir)
    if ok:
        print("\n[✓] All outputs verified. Ready to train.\n")
    else:
        print("\n[✗] Some outputs missing. Check errors above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()