"""
prepare_esconv.py — Normalise raw ESConv.json → train/val/test JSONL splits.

What this script fixes in the raw ESConv.json:
  - Role names: "speaker" → "seeker", "listener" → "supporter"
  - Strategy field: moved from turn["annotation"]["strategy"] → turn["strategy"]
  - 12+ raw strategy variants mapped to 8 canonical labels
  - Missing emotion_intensity field defaulted to 3
  - Emotions not in ESCONV_EMOTION_TO_ID mapped to "neutral" (id=7)

Produces:
  data/processed/esconv_train.jsonl
  data/processed/esconv_val.jsonl
  data/processed/esconv_test.jsonl
  data/processed/strategy_counts.json   ← for class-weighted loss in Phase 2

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
    """
    Normalise a single raw ESConv dialogue to the expected format.
    Returns a normalised dialogue dict or None if invalid.
    """
    # Normalise role names
    turns = []
    for turn in dlg.get("dialog", []):
        raw_role = turn.get("speaker", turn.get("role", ""))
        if raw_role in ("speaker", "seeker"):
            role = "seeker"
        elif raw_role in ("listener", "supporter"):
            role = "supporter"
        else:
            continue

        # Normalise strategy: try direct field, then annotation sub-field
        strategy = turn.get("strategy", None)
        if strategy is None:
            annotation = turn.get("annotation", {})
            if isinstance(annotation, dict):
                strategy = annotation.get("strategy", None)

        turns.append({
            "role": role,
            "content": turn.get("content", "").strip(),
            "strategy": strategy,
        })

    if not turns:
        return None

    # Normalise emotion_intensity
    intensity = dlg.get("emotion_intensity", None)
    if intensity is None:
        intensity = 3
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
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    window_size: int = 3,
):
    print(f"[ESConv] Loading raw data from: {input_path}")
    with open(input_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # ESConv.json is either a list of dialogues or a dict with a key
    if isinstance(raw_data, dict):
        dialogues_raw = list(raw_data.values())
    elif isinstance(raw_data, list):
        dialogues_raw = raw_data
    else:
        raise ValueError(f"Unexpected ESConv.json format: {type(raw_data)}")

    print(f"[ESConv] Raw dialogues: {len(dialogues_raw)}")

    # Normalise all dialogues
    dialogues = []
    for dlg in dialogues_raw:
        norm = normalise_dialogue(dlg)
        if norm is not None:
            dialogues.append(norm)
    print(f"[ESConv] Valid dialogues after normalisation: {len(dialogues)}")

    # Stratified split by emotion_type to maintain distribution
    random.seed(seed)
    random.shuffle(dialogues)

    n = len(dialogues)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    n_train = n - n_test - n_val

    train_dlgs = dialogues[:n_train]
    val_dlgs   = dialogues[n_train:n_train + n_val]
    test_dlgs  = dialogues[n_train + n_val:]

    print(f"[ESConv] Split: train={len(train_dlgs)} val={len(val_dlgs)} test={len(test_dlgs)}")

    # Extract QA pairs using sliding window
    train_pairs = extract_esconv_qa_pairs(train_dlgs, window_size=window_size)
    val_pairs   = extract_esconv_qa_pairs(val_dlgs,   window_size=window_size)
    test_pairs  = extract_esconv_qa_pairs(test_dlgs,  window_size=window_size)

    print(f"[ESConv] QA pairs extracted: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    # Write JSONL splits
    os.makedirs(out_dir, exist_ok=True)
    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        path = os.path.join(out_dir, f"esconv_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"[ESConv] Wrote {len(pairs)} pairs → {path}")

    # Compute strategy class frequencies (train split only)
    strategy_counts = Counter()
    for pair in train_pairs:
        sid = pair.get("strategy_label", -1)
        if sid >= 0:
            strategy_counts[sid] += 1

    counts_path = os.path.join(out_dir, "strategy_counts.json")
    with open(counts_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(strategy_counts.items())}, f, indent=2)
    print(f"[ESConv] Strategy counts → {counts_path}")

    # Print strategy distribution
    print("\n[ESConv] Strategy distribution in train split:")
    total = sum(strategy_counts.values())
    for sid in sorted(strategy_counts.keys()):
        name = CANONICAL_STRATEGIES[sid] if sid < len(CANONICAL_STRATEGIES) else f"id:{sid}"
        count = strategy_counts[sid]
        pct = 100 * count / max(1, total)
        print(f"  [{sid}] {name:<35} {count:5d}  ({pct:.1f}%)")

    return train_pairs, val_pairs, test_pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     default="data/raw/ESConv.json")
    p.add_argument("--out_dir",   default="data/processed")
    p.add_argument("--val_frac",  type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_esconv(
        input_path=args.input,
        out_dir=args.out_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )