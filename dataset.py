
"""
dataset.py — Data Preprocessing and PyTorch Dataset Classes

WHY this file exists:
  The raw EmpatheticDialogues and ESConv datasets cannot be fed to a seq2seq
  model as-is. This module:
    1. Converts EmpatheticDialogues (local Kaggle CSV) into (emotion_label,
       context, response) pairs.
    2. Applies the sliding-window QA extraction algorithm to ESConv multi-turn
       dialogues (Section 2.3 of the spec).
    3. Computes class weights for the strategy loss (Section 2.5).
    4. Exposes PyTorch Dataset objects consumed by both training phases.

Dataset sources (manual download required):
  EmpatheticDialogues:
    Download from Kaggle:
      https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai
    The Kaggle dataset ships as a SINGLE combined CSV file:
      emotion-emotion_69k.csv   (~69 k rows, all splits merged)
    Place it at:
      data/raw/empathetic_dialogues/emotion-emotion_69k.csv

    CSV columns:
      conv_id | utterance_idx | context | prompt | selfeval | tags | utterance | speaker_idx
      - conv_id:       unique dialogue identifier
      - utterance_idx: turn index within the conversation (1-based)
      - context:       emotion label for the whole conversation (e.g. "afraid")
      - utterance:     the spoken text at this turn

  ESConv:
    Clone from GitHub:
      https://github.com/thu-coai/Emotional-Support-Conversation
    Place the JSON file at:
      data/raw/ESConv.json

    Top-level structure: list of dialogue objects.
    Each dialogue object keys:
      emotion_type, problem_type, situation, emotion_intensity, dialog
    Each dialog turn keys:
      role ("seeker" | "supporter"), content, strategy (string or null)
"""

import json
import math
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from tokenizer import (
    HappyBotTokenizer,
    ESCONV_STRATEGY_MAP,
    STRATEGY_TO_ID,
    ESCONV_EMOTION_TO_ID,
    EMPATHETIC_EMOTION_TO_ID,
)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: EmpatheticDialogues Preprocessing  (local Kaggle CSV)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_empathetic_dialogues(
    csv_path: str,
    output_path: str,
    min_token_length: int = 5,
) -> None:
    """
    Read the Kaggle EmpatheticDialogues CSV and produce a flat JSONL file of
    (input, target, emotion_label, strategy_label) records suitable for
    seq2seq training.

    Expected file
    -------------
    The Kaggle dataset (atharvjairath/empathetic-dialogues-facebook-ai) ships
    as a SINGLE combined CSV file named:
        emotion-emotion_69k.csv
    Place it at:
        data/raw/empathetic_dialogues/emotion-emotion_69k.csv

    CSV columns (same as the original Facebook splits):
      conv_id       — unique dialogue identifier
      utterance_idx — integer turn index (1-based); rows are NOT always
                      sorted within a conv_id, so we sort by this field
      context       — emotion label for the whole conversation (e.g. "afraid")
      utterance     — the spoken text at this turn
      (prompt, speaker_idx, selfeval, tags are present but not used)

    Known CSV quirk
    ---------------
    Several rows in the Facebook source CSV have misaligned columns (extra
    commas inside unquoted fields). We use pandas with on_bad_lines="skip"
    (pandas >= 1.3) or error_bad_lines=False (older pandas) to skip those
    ~4 malformed rows without crashing. The loss is negligible (<0.01%).

    Processing steps
    ----------------
    1. Read with pandas, skip malformed rows.
    2. Group by conv_id; sort each group by utterance_idx.
    3. Treat each consecutive pair (turn_i, turn_{i+1}) as (context, response).
    4. Prepend the emotion label as a control token: [emotion_LABEL] <context>.
    5. Filter pairs where either side has fewer than min_token_length words.
    6. Write output JSONL.

    Args:
        csv_path:         Path to the single combined CSV (emotion-emotion_69k.csv
                          or any of the individual train/valid/test splits).
        output_path:      Destination JSONL file path.
        min_token_length: Minimum word count for both sides of a pair.
    """
    import pandas as pd

    csv_path = str(csv_path)
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"EmpatheticDialogues CSV not found: {csv_path}\n"
            "Download from:\n"
            "  https://www.kaggle.com/datasets/atharvjairath/"
            "empathetic-dialogues-facebook-ai\n"
            "Expected file: data/raw/empathetic_dialogues/emotion-emotion_69k.csv"
        )

    # ── Read CSV with pandas (handles the malformed rows in Facebook source) ─
    # on_bad_lines="skip" was added in pandas 1.3; fall back for older installs.
    try:
        df = pd.read_csv(
            csv_path,
            encoding="utf-8",
            on_bad_lines="skip",
            dtype=str,               # read everything as str to avoid mixed-type issues
        )
    except TypeError:
        # pandas < 1.3 uses the deprecated error_bad_lines kwarg
        df = pd.read_csv(
            csv_path,
            encoding="utf-8",
            error_bad_lines=False,
            warn_bad_lines=False,
            dtype=str,
        )

    # Normalise column names (some Kaggle exports add leading/trailing spaces)
    df.columns = [c.strip().lower() for c in df.columns]
    
    required = {"situation", "emotion", "empathetic_dialogues"}
    missing_cols = required - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"CSV is missing expected columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    df = df.dropna(subset=list(required))
    records = []
    skipped_short = 0

    for _, row in df.iterrows():
        emotion_label = str(row["emotion"]).strip().lower()
        situation = str(row["situation"]).strip()
        response = str(row["empathetic_dialogues"]).strip()

        if len(situation.split()) < min_token_length:
            skipped_short += 1
            continue
        if len(response.split()) < min_token_length:
            skipped_short += 1
            continue

        emo_id = EMPATHETIC_EMOTION_TO_ID.get(emotion_label, -1)
        if emo_id == -1:
            continue

        input_text = f"[emotion_{emotion_label}] {situation}"

        records.append({
            "input": input_text,
            "target": response,
            "emotion_label": emo_id,
            "strategy_label": -1,
        })

    # Drop rows where any required column is NaN
    

    

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(
        f"EmpatheticDialogues (Kaggle format) → "
        f"{len(records):,} records written to {output_path} "
        f"(skipped: {skipped_short} short rows)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: ESConv Sliding Window QA Extraction (Section 2.3)
# ─────────────────────────────────────────────────────────────────────────────

def extract_esconv_qa_pairs(
    esconv_json_path: str,
    output_path: str,
    window_size: int = 3,       # k=3 per spec
    max_input_tokens: int = 512,
    tokenizer: Optional[HappyBotTokenizer] = None,
) -> Counter:
    """
    Apply the Sliding Window Context Assembler algorithm (Section 2.3) to
    convert ESConv multi-turn dialogues into flat (input, target) training pairs.

    Algorithm steps (as specified):
      1. For each dialogue D, iterate over every SUPPORTER turn T_i.
      2. Collect the k=3 turns immediately preceding T_i.
      3. Prepend [seeker_emotion_LABEL][intensity_N] metadata tokens.
      4. Prepend situation description only for the FIRST window of each dialogue.
      5. If total tokens > 512, progressively drop the oldest full turn.
      6. Format target as [BOS][strategy_TOKEN] + response_text + [EOS].
      7. Store as (input, target, emotion_id, strategy_id) tuple.

    Returns:
        Counter of strategy label frequencies (used for class-weight computation).
    """
    with open(esconv_json_path) as f:
        dialogues = json.load(f)

    records = []
    strategy_counter = Counter()

    for dlg in dialogues:
        emotion_type = dlg.get("emotion_type", "neutral").lower().strip()
        emotion_id = ESCONV_EMOTION_TO_ID.get(emotion_type, 7)  # default neutral

        intensity = dlg.get("emotion_intensity", 3)
        intensity = max(1, min(5, int(intensity)))

        situation = dlg.get("situation", "").strip()
        turns = dlg.get("dialog", [])

        # Flatten turns to a list of dicts: {role, content, strategy}
        flat_turns = []
        for t in turns:
            flat_turns.append({
                "role":     t.get("role", "seeker"),
                "content":  t.get("content", "").strip(),
                "strategy": t.get("strategy", None),
            })

        is_first_window = True

        # Iterate over every SUPPORTER turn (these become training targets)
        for i, turn in enumerate(flat_turns):
            if turn["role"] != "supporter":
                continue

            raw_strategy = turn.get("strategy") or "Others"
            canonical = ESCONV_STRATEGY_MAP.get(raw_strategy, "other")
            strategy_id = STRATEGY_TO_ID[canonical]
            strategy_counter[canonical] += 1

            response_text = turn["content"]
            if not response_text:
                continue

            # Collect k preceding turns (window)
            preceding = flat_turns[max(0, i - window_size): i]

            # Build context string
            parts = []

            # Metadata prefix: emotion + intensity tokens
            parts.append(
                f"[seeker_emotion_{emotion_type}][intensity_{intensity}]"
            )

            # For the very first window of this dialogue, include situation
            if is_first_window and situation:
                parts.append(f"Situation: {situation}")
                is_first_window = False

            # Add preceding turns with speaker tags
            for prev in preceding:
                role_tag = "[SEEKER]:" if prev["role"] == "seeker" else "[SUPPORTER]:"
                parts.append(f"{role_tag} {prev['content']}")

            input_text = " ".join(parts)

            # Token-budget check: if too long, progressively drop oldest turns
            # WHY: We never truncate mid-sentence (avoids broken context).
            if tokenizer is not None:
                while len(preceding) > 0:
                    ids = tokenizer.encode(input_text, max_length=None)
                    if len(ids) <= max_input_tokens:
                        break
                    # Drop oldest preceding turn and rebuild
                    preceding = preceding[1:]
                    parts_rebuilt = [parts[0]]  # keep metadata prefix
                    if is_first_window is False and situation and len(parts) > 1:
                        # situation was already added, keep it if present
                        pass
                    for prev in preceding:
                        role_tag = "[SEEKER]:" if prev["role"] == "seeker" else "[SUPPORTER]:"
                        parts_rebuilt.append(f"{role_tag} {prev['content']}")
                    input_text = " ".join(parts_rebuilt)

            # Target: [strategy_TOKEN] + response (BOS/EOS added by tokenizer post-processor)
            target_text = f"[strategy_{canonical}] {response_text}"

            records.append({
                "input":          input_text,
                "target":         target_text,
                "emotion_label":  emotion_id,
                "strategy_label": strategy_id,
            })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"ESConv → {len(records)} QA pairs written to {output_path}")
    print(f"Strategy distribution: {dict(strategy_counter)}")
    return strategy_counter


# ─────────────────────────────────────────────────────────────────────────────
# Class weight computation (Section 2.5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_strategy_class_weights(
    strategy_counter: Counter,
    num_classes: int = 8,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for the strategy cross-entropy loss.

    WHY: ESConv has severe strategy imbalance (Questions ~20.7% vs
    Self-Disclosure ~5%). Without weighting, the model collapses to always
    predicting the majority class.

    Formula: weight_c = total_samples / (num_classes * count_c)
    This is the standard sklearn 'balanced' weighting formula.

    Returns:
        Tensor of shape (num_classes,) on `device`.
    """
    total = sum(strategy_counter.values())
    weights = torch.zeros(num_classes)
    for key, canonical in STRATEGY_TO_ID.items():
        count = strategy_counter.get(key, 1)  # avoid div/0
        weights[canonical] = total / (num_classes * count)

    # Normalize so weights sum to num_classes (keeps loss scale stable)
    weights = weights / weights.mean()
    print(f"Strategy class weights: {weights.tolist()}")
    return weights.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Train/Val/Test Split (stratified by strategy_label for ESConv)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_split(
    jsonl_path: str,
    train_path: str,
    val_path: str,
    test_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """
    Stratified 80/10/10 split by strategy_label as specified in Section 4.4.
    For EmpatheticDialogues (strategy_label=-1), falls back to random split.
    """
    random.seed(seed)

    with open(jsonl_path) as f:
        records = [json.loads(line) for line in f]

    # Group by strategy_label
    buckets: Dict[int, List] = {}
    for rec in records:
        sl = rec.get("strategy_label", -1)
        buckets.setdefault(sl, []).append(rec)

    train_recs, val_recs, test_recs = [], [], []
    for sl, recs in buckets.items():
        random.shuffle(recs)
        n = len(recs)
        n_train = math.floor(n * train_ratio)
        n_val = math.floor(n * val_ratio)
        train_recs.extend(recs[:n_train])
        val_recs.extend(recs[n_train: n_train + n_val])
        test_recs.extend(recs[n_train + n_val:])

    def write(recs, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    write(train_recs, train_path)
    write(val_recs, val_path)
    write(test_recs, test_path)
    print(f"Split: {len(train_recs)} train | {len(val_recs)} val | {len(test_recs)} test")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HappyBotDataset(Dataset):
    """
    PyTorch Dataset that reads the preprocessed JSONL files and returns
    tokenized tensors for encoder and decoder.

    Teacher forcing layout (Section 4, Step 4):
      • decoder_input_ids  = [BOS, strategy_token] + target_tokens[:-1]
      • decoder_target_ids = target_tokens[1:] + [EOS]

    Both are shifted by one position so the model predicts the NEXT token
    at every position, which is standard seq2seq teacher forcing.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: HappyBotTokenizer,
        max_src_len: int = 512,
        max_tgt_len: int = 128,
        phase: int = 1,         # 1 = EmpatheticDialogues, 2 = ESConv
    ):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.phase = phase

        with open(jsonl_path) as f:
            self.records = [json.loads(line) for line in f]

        print(f"Dataset loaded: {len(self.records)} records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]

        # ── Encoder input ─────────────────────────────────────────────────
        # CLS token prepended manually so the encoder can extract a
        # classification vector from position 0.
        enc_ids = [self.tokenizer.cls_id] + self.tokenizer.encode(
            rec["input"], max_length=self.max_src_len - 1
        )
        enc_ids = enc_ids[: self.max_src_len]

        # ── Decoder target (what we want to predict) ──────────────────────
        tgt_full = self.tokenizer.encode(rec["target"], max_length=self.max_tgt_len)
        # tgt_full = [BOS, t1, t2, ..., tn, EOS]

        # decoder_input  = [BOS, t1, ..., t_{n-1}]  (feed-in)
        # decoder_target = [t1, ..., tn, EOS]         (supervision signal)
        dec_input = tgt_full[:-1]    # drop last token
        dec_target = tgt_full[1:]    # shift left by 1

        dec_input  = dec_input[: self.max_tgt_len]
        dec_target = dec_target[: self.max_tgt_len]

        # ── Labels ────────────────────────────────────────────────────────
        emotion_label  = rec.get("emotion_label", -1)
        strategy_label = rec.get("strategy_label", -1)

        return {
            "encoder_ids":    torch.tensor(enc_ids,    dtype=torch.long),
            "decoder_input":  torch.tensor(dec_input,  dtype=torch.long),
            "decoder_target": torch.tensor(dec_target, dtype=torch.long),
            "emotion_label":  torch.tensor(emotion_label,  dtype=torch.long),
            "strategy_label": torch.tensor(strategy_label, dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Custom collate function: dynamic padding to batch-max length
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Dynamic padding collate function.

    WHY dynamic padding: Padding all sequences to a global max_len wastes
    compute on short sequences. Instead we pad to the maximum length within
    each batch. This is critical for training speed on variable-length
    dialogue data.

    Returns:
        Dictionary of batched tensors including attention masks.
    """
    def pad_sequence(seqs: List[torch.Tensor], pad_val: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(s.size(0) for s in seqs)
        padded  = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        mask    = torch.zeros(len(seqs), max_len, dtype=torch.bool)  # True = pad position
        for i, s in enumerate(seqs):
            padded[i, :s.size(0)] = s
            mask[i, s.size(0):]   = True
        return padded, mask

    enc_ids_list   = [item["encoder_ids"]    for item in batch]
    dec_in_list    = [item["decoder_input"]  for item in batch]
    dec_tgt_list   = [item["decoder_target"] for item in batch]

    enc_padded, enc_mask = pad_sequence(enc_ids_list, pad_id)
    dec_in_padded, dec_in_mask = pad_sequence(dec_in_list, pad_id)
    dec_tgt_padded, _ = pad_sequence(dec_tgt_list, -100)  # -100 = ignore_index for loss

    return {
        "encoder_ids":     enc_padded,         # (B, S)
        "encoder_mask":    enc_mask,           # (B, S) True = pad
        "decoder_input":   dec_in_padded,      # (B, T)
        "decoder_mask":    dec_in_mask,        # (B, T) True = pad
        "decoder_target":  dec_tgt_padded,     # (B, T) -100 at pad positions
        "emotion_label":   torch.stack([item["emotion_label"]  for item in batch]),
        "strategy_label":  torch.stack([item["strategy_label"] for item in batch]),
    }


def build_dataloader(
    dataset: HappyBotDataset,
    batch_size: int,
    pad_id: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a DataLoader with the custom collate function.

    pin_memory should only be True for CUDA devices; pass False for MPS/CPU
    to avoid the PyTorch warning about unsupported pin_memory on MPS.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
        pin_memory=pin_memory,
    )
