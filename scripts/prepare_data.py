"""
scripts/prepare_data.py — Full Data Preparation Pipeline

Runs all preprocessing steps in order:
  1. Preprocess EmpatheticDialogues (local Kaggle CSVs) →
         data/processed/empathetic_{train,val,test}.jsonl
  2. Extract ESConv QA pairs (local GitHub JSON) →
         data/processed/esconv_{train,val,test}.jsonl
  3. Train BPE tokenizer on combined corpus →
         data/tokenizer/
  4. Save strategy class counts for Phase 2 class weighting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANUAL DOWNLOAD REQUIRED BEFORE RUNNING THIS SCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. EmpatheticDialogues (Kaggle CSV):
       URL  : https://www.kaggle.com/datasets/atharvjairath/
               empathetic-dialogues-facebook-ai
       File : emotion-emotion_69k.csv  (single combined file, ~69k rows)
       Place: data/raw/empathetic_dialogues/emotion-emotion_69k.csv

  2. ESConv (GitHub JSON):
       URL  : https://github.com/thu-coai/Emotional-Support-Conversation
       File : ESConv.json   (in the repo's data/ folder)
       Place: data/raw/ESConv.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run command (from project root):
  python scripts/prepare_data.py

All paths have sensible defaults matching the layout above.
Override them if you stored files elsewhere:

  python scripts/prepare_data.py \\
    --empathetic_csv  data/raw/empathetic_dialogues/emotion-emotion_69k.csv \\
    --esconv_path     data/raw/ESConv.json \\
    --tokenizer_dir   data/tokenizer \\
    --processed_dir   data/processed
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make project root importable regardless of where the script is invoked from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer import train_tokenizer, HappyBotTokenizer
from dataset import (
    preprocess_empathetic_dialogues,
    extract_esconv_qa_pairs,
    stratified_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_raw_files(empathetic_csv: str, esconv_path: str) -> None:
    """
    Fail early with a clear error message if any expected raw file is missing.
    Much friendlier than a cryptic FileNotFoundError mid-pipeline.
    """
    missing = []

    if not Path(empathetic_csv).exists():
        missing.append(empathetic_csv)

    if not Path(esconv_path).exists():
        missing.append(esconv_path)

    if missing:
        print("\n" + "=" * 60)
        print("ERROR: The following raw data files were not found:")
        for m in missing:
            print(f"  x  {m}")
        print()
        print("EmpatheticDialogues: download emotion-emotion_69k.csv from")
        print("  https://www.kaggle.com/datasets/atharvjairath/")
        print("  empathetic-dialogues-facebook-ai")
        print("  Place at: data/raw/empathetic_dialogues/emotion-emotion_69k.csv")
        print()
        print("ESConv: download ESConv.json from")
        print("  https://github.com/thu-coai/Emotional-Support-Conversation")
        print("  Place at: data/raw/ESConv.json")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("All raw data files found.")


def collect_corpus_texts(processed_dir: str, corpus_file: str) -> None:
    """
    Concatenate every input and target string from the processed JSONL files
    into a single plain-text file (one sentence per line) for BPE training.

    WHY both train splits: the tokenizer should see the full vocabulary of
    both datasets so that no in-vocabulary words are mapped to [UNK] during
    the actual training runs.
    """
    texts = []
    for fname in ["empathetic_train.jsonl", "esconv_train.jsonl"]:
        path = os.path.join(processed_dir, fname)
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — skipping for corpus.")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append(rec["input"].strip())
                texts.append(rec["target"].strip())

    with open(corpus_file, "w", encoding="utf-8") as f:
        for t in texts:
            if t:
                f.write(t + "\n")

    print(f"  Corpus assembled: {len(texts):,} lines -> {corpus_file}")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Happy-Bot data preparation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--empathetic_csv",
        default="data/raw/empathetic_dialogues/emotion-emotion_69k.csv",
        help=(
            "Path to the single Kaggle EmpatheticDialogues CSV file "
            "(emotion-emotion_69k.csv).  "
            "Default: data/raw/empathetic_dialogues/emotion-emotion_69k.csv"
        ),
    )
    p.add_argument(
        "--esconv_path",
        default="data/raw/ESConv.json",
        help=(
            "Path to ESConv.json from the GitHub repo.  "
            "Default: data/raw/ESConv.json"
        ),
    )
    p.add_argument("--tokenizer_dir",  default="data/tokenizer")
    p.add_argument("--processed_dir",  default="data/processed")
    p.add_argument("--vocab_size",     type=int, default=10_000)
    p.add_argument(
        "--window_size",
        type=int,
        default=3,
        help="Sliding-window size k for ESConv QA extraction (default: 3).",
    )
    p.add_argument(
        "--skip_empathetic",
        action="store_true",
        help="Skip EmpatheticDialogues preprocessing (use if already done).",
    )
    p.add_argument(
        "--skip_esconv",
        action="store_true",
        help="Skip ESConv preprocessing (use if already done).",
    )
    p.add_argument(
        "--skip_tokenizer",
        action="store_true",
        help="Skip tokenizer training (use if tokenizer already exists).",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve absolute paths so relative-path mistakes are caught early
    empathetic_csv = str(Path(args.empathetic_csv).resolve())
    esconv_path    = str(Path(args.esconv_path).resolve())
    processed_dir  = str(Path(args.processed_dir).resolve())
    tokenizer_dir  = str(Path(args.tokenizer_dir).resolve())

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(tokenizer_dir,  exist_ok=True)

    print("\n" + "=" * 60)
    print("  Happy-Bot -- Data Preparation Pipeline")
    print("=" * 60)
    print(f"  EmpatheticDialogues CSV  : {empathetic_csv}")
    print(f"  ESConv JSON              : {esconv_path}")
    print(f"  Processed output         : {processed_dir}/")
    print(f"  Tokenizer output         : {tokenizer_dir}/")
    print("=" * 60 + "\n")

    # ── Pre-flight check ───────────────────────────────────────────────────
    skip_all_raw = args.skip_empathetic and args.skip_esconv
    if not skip_all_raw:
        _check_raw_files(empathetic_csv, esconv_path)

    # ======================================================================
    # Step 1 -- EmpatheticDialogues  (single Kaggle CSV -> JSONL)
    # ======================================================================
    if not args.skip_empathetic:
        print("-" * 60)
        print("Step 1 -- EmpatheticDialogues preprocessing")
        print("-" * 60)

        # emotion-emotion_69k.csv is already the full combined dataset.
        # Preprocess once, then do stratified 80/10/10 split by emotion_label
        # as required by Section 4.4 of the spec.
        empathetic_full = os.path.join(processed_dir, "empathetic_full.jsonl")

        preprocess_empathetic_dialogues(
            csv_path=empathetic_csv,
            output_path=empathetic_full,
        )

        with open(empathetic_full) as f:
            n = sum(1 for _ in f)
        print(f"  Full JSONL: {n:,} records -> {empathetic_full}")

        # Stratified 80/10/10 split by emotion_label
        stratified_split(
            jsonl_path=empathetic_full,
            train_path=os.path.join(processed_dir, "empathetic_train.jsonl"),
            val_path=os.path.join(processed_dir,   "empathetic_val.jsonl"),
            test_path=os.path.join(processed_dir,  "empathetic_test.jsonl"),
        )
        print()
    else:
        print("Step 1 -- EmpatheticDialogues: SKIPPED\n")

    # ======================================================================
    # Step 2 -- ESConv  (GitHub JSON -> sliding-window QA pairs -> JSONL)
    # ======================================================================
    if not args.skip_esconv:
        print("-" * 60)
        print("Step 2 -- ESConv sliding-window QA extraction")
        print("-" * 60)

        esconv_full = os.path.join(processed_dir, "esconv_full.jsonl")
        counts_path = os.path.join(processed_dir, "strategy_counts.json")

        # tokenizer=None on first run: token-budget enforcement uses word
        # count as an approximation (accurate enough for window trimming).
        strategy_counter = extract_esconv_qa_pairs(
            esconv_json_path=esconv_path,
            output_path=esconv_full,
            window_size=args.window_size,
            tokenizer=None,
        )

        # Save strategy counts for Phase 2 class-weight computation
        with open(counts_path, "w") as f:
            json.dump(dict(strategy_counter), f, indent=2)
        print(f"  Strategy counts saved -> {counts_path}")

        # Stratified 80/10/10 split by strategy_label
        stratified_split(
            jsonl_path=esconv_full,
            train_path=os.path.join(processed_dir, "esconv_train.jsonl"),
            val_path=os.path.join(processed_dir,   "esconv_val.jsonl"),
            test_path=os.path.join(processed_dir,  "esconv_test.jsonl"),
        )
        print()
    else:
        print("Step 2 -- ESConv: SKIPPED\n")

    # ======================================================================
    # Step 3 -- Train BPE tokenizer
    # ======================================================================
    if not args.skip_tokenizer:
        print("-" * 60)
        print("Step 3 -- BPE tokenizer training")
        print("-" * 60)

        corpus_file = os.path.join(processed_dir, "corpus.txt")
        collect_corpus_texts(processed_dir, corpus_file)

        train_tokenizer(
            corpus_files=[corpus_file],
            save_dir=tokenizer_dir,
            vocab_size=args.vocab_size,
        )
        print()
    else:
        print("Step 3 -- Tokenizer training: SKIPPED\n")

    # ======================================================================
    # Step 4 -- Round-trip verification
    # ======================================================================
    print("-" * 60)
    print("Step 4 -- Tokenizer round-trip verification")
    print("-" * 60)

    tok = HappyBotTokenizer(tokenizer_dir)
    samples = [
        "[emotion_anxious] I have a big presentation tomorrow and I can't focus.",
        "[seeker_emotion_anxiety][intensity_4] [SEEKER]: I haven't slept in days.",
        "[strategy_reflection] It sounds like you're really overwhelmed right now.",
    ]
    for s in samples:
        ids     = tok.encode(s)
        ok_flag = "OK" if len(ids) > 1 else "FAIL"
        print(f"  [{ok_flag}]  ids={len(ids):3d}  '{s[:58]}'")

    # ======================================================================
    # Summary
    # ======================================================================
    print()
    print("=" * 60)
    print("  Data preparation complete!")
    print("=" * 60)

    def _count_lines(path):
        try:
            with open(path) as f:
                return sum(1 for _ in f)
        except FileNotFoundError:
            return "missing"

    report_files = {
        "empathetic_train.jsonl": os.path.join(processed_dir, "empathetic_train.jsonl"),
        "empathetic_val.jsonl":   os.path.join(processed_dir, "empathetic_val.jsonl"),
        "empathetic_test.jsonl":  os.path.join(processed_dir, "empathetic_test.jsonl"),
        "esconv_train.jsonl":     os.path.join(processed_dir, "esconv_train.jsonl"),
        "esconv_val.jsonl":       os.path.join(processed_dir, "esconv_val.jsonl"),
        "esconv_test.jsonl":      os.path.join(processed_dir, "esconv_test.jsonl"),
        "strategy_counts.json":   os.path.join(processed_dir, "strategy_counts.json"),
    }
    for name, path in report_files.items():
        n      = _count_lines(path)
        exists = "OK" if Path(path).exists() else "MISSING"
        print(f"  [{exists}]  {name:<30}  {n}")

    print(f"\n  Tokenizer vocab size : {tok.vocab_size:,}")
    print(f"  Tokenizer directory  : {tokenizer_dir}/")
    print(f"  Processed directory  : {processed_dir}/")
    print()
    print("Next step -> run Phase 1 training:")
    print("  python train_phase1.py")
    print()


if __name__ == "__main__":
    main()
