"""
prepare_esconv.py — Normalise raw ESConv.json → train/val/test JSONL splits.

FIXES vs original:
  1. Strategy map now covers all 12 raw ESConv strategy strings (see dataset.py).
  2. strategy_counts.json now ALWAYS writes all 8 class keys (0–7), even when
     count=0, so compute_strategy_weights() always gets a complete dict and
     never silently assigns weight=1.0 to unobserved classes.
  3. Minimum content length filter: skip supporter turns with < 3 words to
     reduce noise (single-word acks like "okay" and "hello" warp the loss).
  4. Deduplication: identical (input, target) pairs removed from train split.

Usage:
  python prepare_esconv.py
  python prepare_esconv.py --input data/raw/ESConv.json --out_dir data/processed
"""

import argparse
import json
import os
import random
from collections import Counter

from dataset import extract_esconv_qa_pairs, CANONICAL_STRATEGIES, STRATEGY_TO_ID


def normalise_dialogue(dlg: dict) -> dict:
    """Normalise a single raw ESConv dialogue. Returns None if invalid."""
    turns = []
    for turn in dlg.get("dialog", []):
        raw_role = turn.get("speaker", turn.get("role", ""))
        if raw_role in ("speaker", "seeker"):
            role = "seeker"
        elif raw_role in ("listener", "supporter"):
            role = "supporter"
        else:
            continue

        # Strategy: try direct field first, then annotation sub-dict
        strategy = turn.get("strategy", None)
        if strategy is None:
            ann = turn.get("annotation", {})
            if isinstance(ann, dict):
                strategy = ann.get("strategy", None)

        turns.append({
            "role":     role,
            "content":  turn.get("content", "").strip(),
            "strategy": strategy,
        })

    if not turns:
        return None

    intensity = dlg.get("emotion_intensity", 3)
    try:
        intensity = int(intensity)
    except (ValueError, TypeError):
        intensity = 3

    return {
        "emotion_type":      dlg.get("emotion_type", "neutral").lower(),
        "problem_type":      dlg.get("problem_type", ""),
        "situation":         dlg.get("situation", ""),
        "emotion_intensity": intensity,
        "dialog":            turns,
    }


def prepare_esconv(
    input_path: str,
    out_dir: str,
    val_frac: float  = 0.15,
    test_frac: float = 0.15,
    seed: int        = 42,
    window_size: int = 3,
    min_target_words: int = 3,
):
    print(f"[ESConv] Loading raw data from: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    dialogues_raw = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
    print(f"[ESConv] Raw dialogues: {len(dialogues_raw)}")

    dialogues = [d for d in (normalise_dialogue(r) for r in dialogues_raw) if d]
    print(f"[ESConv] Valid dialogues after normalisation: {len(dialogues)}")

    random.seed(seed)
    random.shuffle(dialogues)

    n       = len(dialogues)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_test - n_val

    train_dlgs = dialogues[:n_train]
    val_dlgs   = dialogues[n_train:n_train + n_val]
    test_dlgs  = dialogues[n_train + n_val:]
    print(f"[ESConv] Split: train={len(train_dlgs)} val={len(val_dlgs)} test={len(test_dlgs)}")

    train_pairs = extract_esconv_qa_pairs(train_dlgs, window_size=window_size)
    val_pairs   = extract_esconv_qa_pairs(val_dlgs,   window_size=window_size)
    test_pairs  = extract_esconv_qa_pairs(test_dlgs,  window_size=window_size)

    # FIX 3: Filter out very short targets (noise)
    def filter_pairs(pairs):
        return [p for p in pairs if len(p["target"].split()) >= min_target_words]

    train_pairs = filter_pairs(train_pairs)
    val_pairs   = filter_pairs(val_pairs)
    test_pairs  = filter_pairs(test_pairs)

    # FIX 4: Deduplicate train split on (input, target)
    seen = set()
    deduped = []
    for p in train_pairs:
        key = (p["input"], p["target"])
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    print(f"[ESConv] After dedup: train {len(train_pairs)} → {len(deduped)}")
    train_pairs = deduped

    print(f"[ESConv] Final pairs: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    # Write JSONL splits
    os.makedirs(out_dir, exist_ok=True)
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        path = os.path.join(out_dir, f"esconv_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"[ESConv] Wrote {len(pairs)} pairs → {path}")

    # FIX 2: strategy_counts.json — always write ALL 8 class keys.
    # The original only wrote observed classes, causing weight=1.0 for any
    # missing class in compute_strategy_weights().
    strategy_counts = Counter(p["strategy_label"] for p in train_pairs if p["strategy_label"] >= 0)
    counts_dict = {str(i): strategy_counts.get(i, 0) for i in range(len(CANONICAL_STRATEGIES))}

    counts_path = os.path.join(out_dir, "strategy_counts.json")
    with open(counts_path, "w") as f:
        json.dump(counts_dict, f, indent=2)
    print(f"[ESConv] Strategy counts → {counts_path}")

    print("\n[ESConv] Strategy distribution in train split:")
    total = sum(strategy_counts.values())
    for sid in range(len(CANONICAL_STRATEGIES)):
        name  = CANONICAL_STRATEGIES[sid]
        count = strategy_counts.get(sid, 0)
        pct   = 100 * count / max(1, total)
        print(f"  [{sid}] {name:<35} {count:5d}  ({pct:.1f}%)")

    return train_pairs, val_pairs, test_pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",             default="data/raw/ESConv.json")
    p.add_argument("--out_dir",           default="data/processed")
    p.add_argument("--val_frac",          type=float, default=0.15)
    p.add_argument("--test_frac",         type=float, default=0.15)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--min_target_words",  type=int,   default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_esconv(
        input_path       = args.input,
        out_dir          = args.out_dir,
        val_frac         = args.val_frac,
        test_frac        = args.test_frac,
        seed             = args.seed,
        min_target_words = args.min_target_words,
    )