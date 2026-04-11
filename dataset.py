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
"""

import json
import os
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


ESCONV_STRATEGY_MAP = {
    # Raw ESConv strategy strings → canonical 8-class labels
    "Question":                     "question",
    "Restatement or Paraphrasing":  "restatement",
    "Reflection of Feelings":       "restatement",       # merged
    "Acknowledgement and Emphasis": "affirmation_and_reassurance",
    "Affirmation and Reassurance":  "affirmation_and_reassurance",
    "Providing Suggestions":        "suggestion",
    "Providing Suggestions or Information": "suggestion",
    "Self-disclosure":              "self_disclosure",
    "Others experiences":           "others_experiences",
    "Information":                  "information",
    "Transition":                   "transition_to_problem",
    "Others":                       "transition_to_problem",  # catch-all
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
    "anxiety": 0, "depression": 1, "sadness": 2, "anger": 3,
    "fear": 4, "disgust": 5, "jealousy": 6, "neutral": 7,
}


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
        self.tokenizer = tokenizer
        self.phase = phase
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Special token ids
        self.pad_id      = tokenizer.token_to_id("[PAD]")   or 0
        self.bos_id      = tokenizer.token_to_id("[BOS]")   or 1
        self.eos_id      = tokenizer.token_to_id("[EOS]")   or 2
        self.unk_id      = tokenizer.token_to_id("[UNK]")   or 3

        # Strategy token ids (for decoder seed injection)
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
                obj = json.loads(line)
                self.samples.append(obj)

        print(f"  [Dataset] Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # ── Encoder input ───────────────────────────────────────────────
        src_text = sample["input"]
        enc = self.tokenizer.encode(src_text)
        src_ids = enc.ids[: self.max_src_len]   # truncate if too long

        # ── Strategy token for decoder seed ─────────────────────────────
        strategy_label = int(sample.get("strategy_label", -1))
        if strategy_label >= 0 and strategy_label < len(CANONICAL_STRATEGIES):
            strat_name = CANONICAL_STRATEGIES[strategy_label]
            strat_tok_id = self.strategy_token_ids.get(strat_name)
        else:
            strat_tok_id = None

        # ── Decoder input (teacher forcing) ─────────────────────────────
        # Format: [BOS] [strategy_token?] w1 w2 ... w_{n-1}
        tgt_text = sample["target"]
        tgt_enc = self.tokenizer.encode(tgt_text)
        tgt_ids = tgt_enc.ids[: self.max_tgt_len]

        seed = [self.bos_id]
        if strat_tok_id is not None:
            seed.append(strat_tok_id)

        # decoder_input:  [BOS (strat)] + target_tokens[:-1]
        # decoder_target: target_tokens + [EOS]
        # This is the teacher-forcing (shifted) setup.
        decoder_input  = seed + tgt_ids[:-1] if tgt_ids else seed
        decoder_target = tgt_ids + [self.eos_id]

        # Truncate decoder sequences
        decoder_input  = decoder_input[: self.max_tgt_len]
        decoder_target = decoder_target[: self.max_tgt_len]

        emotion_label = int(sample.get("emotion_label", -1))

        return {
            "encoder_ids":       torch.tensor(src_ids,       dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
            "decoder_target_ids": torch.tensor(decoder_target, dtype=torch.long),
            "emotion_label":     torch.tensor(emotion_label, dtype=torch.long),
            "strategy_label":    torch.tensor(strategy_label, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_id: int = 0) -> dict:
    """
    Dynamic padding collate function.
    Pads all sequences in a batch to the longest sequence in that batch.
    Decoder target positions beyond the true target (i.e., PAD positions) are
    replaced with -100 so cross-entropy ignores them.
    """
    def pad_sequence(seqs: list[torch.Tensor], pad_val: int) -> torch.Tensor:
        max_len = max(s.size(0) for s in seqs)
        padded = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : s.size(0)] = s
        return padded

    encoder_ids       = pad_sequence([b["encoder_ids"]       for b in batch], pad_id)
    decoder_input_ids = pad_sequence([b["decoder_input_ids"] for b in batch], pad_id)

    # Decoder targets: pad with -100 (CrossEntropyLoss ignore_index)
    decoder_target_ids = pad_sequence([b["decoder_target_ids"] for b in batch], -100)

    emotion_labels  = torch.stack([b["emotion_label"]  for b in batch])
    strategy_labels = torch.stack([b["strategy_label"] for b in batch])

    return {
        "encoder_ids":        encoder_ids,
        "decoder_input_ids":  decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
        "emotion_label":      emotion_labels,
        "strategy_label":     strategy_labels,
    }


# ── ESConv QA Pair Extraction ─────────────────────────────────────────────────

def extract_esconv_qa_pairs(
    dialogues: list[dict],
    window_size: int = 3,
    max_tokens: int = 512,
) -> list[dict]:
    """
    Convert ESConv multi-turn dialogues into flat (input, target) training pairs
    using a sliding window of k=3 prior turns.
    
    This is the core ESConv preprocessing function.
    Called by prepare_esconv.py.

    Each extracted pair:
      input:  [CLS][seeker_emotion_X][intensity_N] situation + window of k prior turns
      target: supporter response (the "answer" to predict)
      emotion_label: int
      strategy_label: int
    """
    pairs = []

    for dlg in dialogues:
        emotion_type = dlg.get("emotion_type", "neutral").lower()
        emotion_label = ESCONV_EMOTION_TO_ID.get(emotion_type, 7)

        intensity = int(dlg.get("emotion_intensity", 3))
        situation = dlg.get("situation", "")
        turns = dlg.get("dialog", [])

        for i, turn in enumerate(turns):
            role = turn.get("role", turn.get("speaker", ""))
            # Normalize role names
            if role in ("listener", "supporter"):
                role = "supporter"
            elif role in ("speaker", "seeker"):
                role = "seeker"
            else:
                continue

            if role != "supporter":
                continue

            content = turn.get("content", "").strip()
            if not content:
                continue

            # Get strategy label
            raw_strategy = turn.get("strategy", None)
            if raw_strategy is None:
                annotation = turn.get("annotation", {})
                if isinstance(annotation, dict):
                    raw_strategy = annotation.get("strategy", None)

            canonical = ESCONV_STRATEGY_MAP.get(raw_strategy, "question")
            strategy_label = STRATEGY_TO_ID.get(canonical, 0)

            # Assemble context window (k prior turns)
            window_turns = []
            for j in range(max(0, i - window_size), i):
                t = turns[j]
                t_role = t.get("role", t.get("speaker", ""))
                t_role = "supporter" if t_role in ("listener", "supporter") else "seeker"
                t_content = t.get("content", "").strip()
                if t_content:
                    window_turns.append(f"[{t_role.upper()}]: {t_content}")

            # Build the full input string
            metadata = f"[CLS][seeker_emotion_{emotion_type.upper()}][intensity_{intensity}]"
            if i == 0 and situation:
                situation_prefix = f"[SITUATION]: {situation} "
            else:
                situation_prefix = ""
            context_str = " ".join(window_turns)
            input_text = f"{metadata} {situation_prefix}{context_str}".strip()

            pairs.append({
                "input":          input_text,
                "target":         content,
                "emotion_label":  emotion_label,
                "strategy_label": strategy_label,
            })

    return pairs