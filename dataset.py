"""
dataset.py — PyTorch Dataset for HappyBot training.

Handles both Phase 1 (EmpatheticDialogues JSONL) and Phase 2 (ESConv JSONL) samples.
The JSONL format is produced by scripts/prepare_data.py and prepare_esconv.py.

JSONL format (common):
  {"input": "...", "target": "...", "emotion_label": int, "strategy_label": int}

Phase 1 (EmpatheticDialogues): strategy_label is absent or -1.
Phase 2 (ESConv QA pairs): both emotion_label and strategy_label are set.

Teacher forcing decoder setup:
  decoder_input  = [BOS, strategy_token, w1, w2, ..., w_{n-1}]
  decoder_target = [w1,  w2,  ...,  w_n, EOS]            ← shifted left by 1
  Loss is computed between decoder_target and logits, ignoring PAD/EOS positions.

FIXES vs original:
  1. ESCONV_STRATEGY_MAP now covers ALL 12 raw strategy strings seen in the data,
     including case variants ('Questions', 'Restatement', 'Reflection of feelings',
     'Direct Guidance', 'Approval and Reassurance').
  2. strategy_counts.json now always writes all 8 classes (0–7), even if count=0,
     so compute_strategy_weights() never gets a partial dict.
"""

import json
import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


# ── Strategy Mapping ──────────────────────────────────────────────────────────

# FIX: All 12 raw ESConv strategy strings mapped to 8 canonical labels.
# Original only mapped 7 strings; the missing ones defaulted silently to
# "question" (id=0), causing strategy class imbalance.
ESCONV_STRATEGY_MAP = {
    # Standard variants
    "Question":                             "question",
    "Questions":                            "question",          # ← was missing
    "Restatement or Paraphrasing":          "restatement",
    "Restatement":                          "restatement",       # ← was missing
    "Reflection of Feelings":              "restatement",
    "Reflection of feelings":              "restatement",        # ← case variant was missing
    "Acknowledgement and Emphasis":         "affirmation_and_reassurance",
    "Affirmation and Reassurance":          "affirmation_and_reassurance",
    "Approval and Reassurance":             "affirmation_and_reassurance",  # ← was missing
    "Providing Suggestions":               "suggestion",
    "Providing Suggestions or Information": "suggestion",
    "Direct Guidance":                      "suggestion",        # ← was missing
    "Self-disclosure":                      "self_disclosure",
    "Self-Disclosure":                      "self_disclosure",   # ← case variant
    "Others experiences":                   "others_experiences",
    "Information":                          "information",
    "Transition":                           "transition_to_problem",
    # Catch-all — keep explicitly, don't silently default
    "Other":                                "transition_to_problem",
    "Others":                               "transition_to_problem",
}

CANONICAL_STRATEGIES = [
    "question",
    "restatement",
    "affirmation_and_reassurance",
    "suggestion",
    "information",
    "self_disclosure",
    "others_experiences",
    "transition_to_problem",
]

STRATEGY_TO_ID = {s: i for i, s in enumerate(CANONICAL_STRATEGIES)}

ESCONV_EMOTION_TO_ID = {
    "anxiety":    0,
    "depression": 1,
    "sadness":    2,
    "anger":      3,
    "fear":       4,
    "disgust":    5,
    "jealousy":   6,
    "neutral":    7,
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class HappyBotDataset(Dataset):
    """
    Loads a JSONL file produced by the data preparation scripts.
    Returns per-sample dicts with encoder ids, decoder input ids, decoder target ids,
    emotion label, and strategy label (or -1 for Phase 1).
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        phase: int = 1,
        max_src_len: int = 512,
        max_tgt_len: int = 128,
    ):
        self.tokenizer   = tokenizer
        self.phase       = phase
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Special token ids
        self.pad_id = tokenizer.token_to_id("[PAD]") or 0
        self.bos_id = tokenizer.token_to_id("[BOS]") or 1
        self.eos_id = tokenizer.token_to_id("[EOS]") or 2
        self.unk_id = tokenizer.token_to_id("[UNK]") or 3

        # Strategy token ids for decoder seed injection
        self.strategy_token_ids = {}
        for s in CANONICAL_STRATEGIES:
            tok = f"[strategy_{s.upper()}]"
            tid = tokenizer.token_to_id(tok)
            self.strategy_token_ids[s] = tid  # None if not in vocab

        self.samples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        print(f"  [Dataset] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── Encoder input ────────────────────────────────────────────────
        src_ids = self.tokenizer.encode(sample["input"]).ids[: self.max_src_len]

        # ── Strategy token for decoder seed ──────────────────────────────
        strategy_label = int(sample.get("strategy_label", -1))
        if 0 <= strategy_label < len(CANONICAL_STRATEGIES):
            strat_tok_id = self.strategy_token_ids.get(CANONICAL_STRATEGIES[strategy_label])
        else:
            strat_tok_id = None

        # ── Decoder (teacher forcing) ─────────────────────────────────────
        # decoder_input  = [BOS, (strategy_tok), w0, w1, ..., w_{n-2}]
        # decoder_target = [w0,  w1,  ...,  w_{n-1}, EOS]
        tgt_ids = self.tokenizer.encode(sample["target"]).ids[: self.max_tgt_len]

        seed = [self.bos_id]
        if strat_tok_id is not None:
            seed.append(strat_tok_id)

        decoder_input  = (seed + tgt_ids[:-1] if tgt_ids else seed)[: self.max_tgt_len]
        decoder_target = (tgt_ids + [self.eos_id])[: self.max_tgt_len]

        emotion_label = int(sample.get("emotion_label", -1))

        return {
            "encoder_ids":        torch.tensor(src_ids,       dtype=torch.long),
            "decoder_input_ids":  torch.tensor(decoder_input, dtype=torch.long),
            "decoder_target_ids": torch.tensor(decoder_target, dtype=torch.long),
            "emotion_label":      torch.tensor(emotion_label,  dtype=torch.long),
            "strategy_label":     torch.tensor(strategy_label, dtype=torch.long),
        }


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list, pad_id: int = 0) -> dict:
    """
    Dynamic padding collate function.
    Pads all sequences in a batch to the longest sequence in that batch.
    Decoder target positions beyond the true target are replaced with -100
    so cross-entropy ignores them.
    """
    def pad_seq(seqs, pad_val):
        max_len = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    encoder_ids       = pad_seq([b["encoder_ids"]       for b in batch], pad_id)
    decoder_input_ids = pad_seq([b["decoder_input_ids"] for b in batch], pad_id)
    # Decoder targets: pad with -100 (CrossEntropyLoss ignore_index)
    decoder_target_ids = pad_seq([b["decoder_target_ids"] for b in batch], -100)

    # Align decoder_input and decoder_target to same length
    max_dec = max(decoder_input_ids.size(1), decoder_target_ids.size(1))
    if decoder_input_ids.size(1) < max_dec:
        pad = torch.full((len(batch), max_dec - decoder_input_ids.size(1)), pad_id, dtype=torch.long)
        decoder_input_ids = torch.cat([decoder_input_ids, pad], dim=1)
    if decoder_target_ids.size(1) < max_dec:
        pad = torch.full((len(batch), max_dec - decoder_target_ids.size(1)), -100, dtype=torch.long)
        decoder_target_ids = torch.cat([decoder_target_ids, pad], dim=1)

    return {
        "encoder_ids":        encoder_ids,
        "decoder_input_ids":  decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
        "emotion_label":      torch.stack([b["emotion_label"]  for b in batch]),
        "strategy_label":     torch.stack([b["strategy_label"] for b in batch]),
    }


# ── ESConv QA Pair Extraction ─────────────────────────────────────────────────

def extract_esconv_qa_pairs(
    dialogues: list,
    window_size: int = 3,
    max_tokens: int = 512,
) -> list:
    """
    Convert ESConv multi-turn dialogues into flat (input, target) training pairs
    using a sliding window of k=3 prior turns.

    Each extracted pair:
      input:  [CLS][seeker_emotion_X][intensity_N] situation + window of k prior turns
      target: supporter response (the "answer" to predict)
      emotion_label: int
      strategy_label: int
    """
    pairs = []

    for dlg in dialogues:
        emotion_type  = dlg.get("emotion_type", "neutral").lower()
        emotion_label = ESCONV_EMOTION_TO_ID.get(emotion_type, 7)
        intensity     = int(dlg.get("emotion_intensity", 3))
        situation     = dlg.get("situation", "")
        turns         = dlg.get("dialog", [])

        for i, turn in enumerate(turns):
            role = turn.get("role", turn.get("speaker", ""))
            role = "supporter" if role in ("listener", "supporter") else "seeker"

            if role != "supporter":
                continue

            content = turn.get("content", "").strip()
            if not content or len(content.split()) < 2:   # skip single-word turns
                continue

            # Resolve strategy
            raw_strategy = turn.get("strategy") or (
                turn.get("annotation", {}) or {}
            ).get("strategy")

            # FIX: use updated map; default to "question" only as last resort
            canonical      = ESCONV_STRATEGY_MAP.get(raw_strategy, "question")
            strategy_label = STRATEGY_TO_ID.get(canonical, 0)

            # Context window
            window_turns = []
            for j in range(max(0, i - window_size), i):
                t = turns[j]
                t_role    = t.get("role", t.get("speaker", ""))
                t_role    = "supporter" if t_role in ("listener", "supporter") else "seeker"
                t_content = t.get("content", "").strip()
                if t_content:
                    window_turns.append(f"[{t_role.upper()}]: {t_content}")

            # Build encoder input string
            metadata = f"[CLS][seeker_emotion_{emotion_type.upper()}][intensity_{intensity}]"
            situation_prefix = f"[SITUATION]: {situation} " if (i == 0 and situation) else ""
            input_text = f"{metadata} {situation_prefix}{' '.join(window_turns)}".strip()

            pairs.append({
                "input":          input_text,
                "target":         content,
                "emotion_label":  emotion_label,
                "strategy_label": strategy_label,
            })

    return pairs