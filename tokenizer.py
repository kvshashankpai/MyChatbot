"""
tokenizer.py — Custom BPE Tokenizer for Happy-Bot

WHY: We train our own tokenizer exclusively on the project corpus
(EmpatheticDialogues + ESConv) so that:
  1. Vocabulary distribution matches our domain (therapeutic language).
  2. All special control tokens (emotion, strategy, intensity) are atomic
     — they are never split by BPE merges.
  3. The embedding matrix stays small (~10K tokens) on limited GPU memory.
"""

import os
import json
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing


# ─────────────────────────────────────────────────────────────────────────────
# Special token definitions
# ─────────────────────────────────────────────────────────────────────────────

# Structural tokens (must be added BEFORE BPE merges)
STRUCTURAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[CLS]", "[SEP]",
                     "[SEEKER]:", "[SUPPORTER]:"]

# Emotion tokens from ESConv
ESCONV_EMOTION_TOKENS = [
    "[seeker_emotion_anxiety]", "[seeker_emotion_sadness]",
    "[seeker_emotion_anger]", "[seeker_emotion_fear]",
    "[seeker_emotion_disgust]", "[seeker_emotion_joy]",
    "[seeker_emotion_surprise]", "[seeker_emotion_neutral]",
]

# All 32 EmpatheticDialogues emotion labels as special tokens
EMPATHETIC_EMOTION_TOKENS = [
    "[emotion_afraid]", "[emotion_angry]", "[emotion_annoyed]",
    "[emotion_anticipating]", "[emotion_anxious]", "[emotion_apprehensive]",
    "[emotion_ashamed]", "[emotion_caring]", "[emotion_confident]",
    "[emotion_content]", "[emotion_devastated]", "[emotion_disappointed]",
    "[emotion_disgusted]", "[emotion_embarrassed]", "[emotion_excited]",
    "[emotion_faithful]", "[emotion_furious]", "[emotion_grateful]",
    "[emotion_guilty]", "[emotion_hopeful]", "[emotion_impressed]",
    "[emotion_jealous]", "[emotion_joyful]", "[emotion_lonely]",
    "[emotion_nostalgic]", "[emotion_prepared]", "[emotion_proud]",
    "[emotion_sad]", "[emotion_sentimental]", "[emotion_surprised]",
    "[emotion_terrified]", "[emotion_trusting]",
]

# Strategy control tokens — prepended to decoder input at generation time
STRATEGY_TOKENS = [
    "[strategy_question]", "[strategy_restatement]",
    "[strategy_reflection]", "[strategy_affirmation]",
    "[strategy_suggestion]", "[strategy_information]",
    "[strategy_selfdisclosure]", "[strategy_other]",
]

# Intensity tokens (ESConv 1-5 scale)
INTENSITY_TOKENS = [f"[intensity_{i}]" for i in range(1, 6)]

ALL_SPECIAL_TOKENS = (
    STRUCTURAL_TOKENS
    + ESCONV_EMOTION_TOKENS
    + EMPATHETIC_EMOTION_TOKENS
    + STRATEGY_TOKENS
    + INTENSITY_TOKENS
)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical string → ID mappings for strategy and emotion heads
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_TO_ID = {
    "question":        0,
    "restatement":     1,
    "reflection":      2,
    "affirmation":     3,
    "suggestion":      4,
    "information":     5,
    "selfdisclosure":  6,
    "other":           7,
}
ID_TO_STRATEGY = {v: k for k, v in STRATEGY_TO_ID.items()}

# Map ESConv raw strategy strings → canonical keys
ESCONV_STRATEGY_MAP = {
    "Question":                      "question",
    "Restatement or Paraphrasing":   "restatement",
    "Reflection of Feelings":        "reflection",
    "Affirmation and Reassurance":   "affirmation",
    "Providing Suggestions":         "suggestion",
    "Information":                   "information",
    "Self-disclosure":               "selfdisclosure",
    "Others":                        "other",
}

# ESConv emotion → ID (8 coarse classes)
ESCONV_EMOTION_TO_ID = {
    "anxiety": 0, "sadness": 1, "anger": 2, "fear": 3,
    "disgust": 4, "joy": 5, "surprise": 6, "neutral": 7,
}

# EmpatheticDialogues 32-class → ID
EMPATHETIC_EMOTION_TO_ID = {
    "afraid": 0, "angry": 1, "annoyed": 2, "anticipating": 3,
    "anxious": 4, "apprehensive": 5, "ashamed": 6, "caring": 7,
    "confident": 8, "content": 9, "devastated": 10, "disappointed": 11,
    "disgusted": 12, "embarrassed": 13, "excited": 14, "faithful": 15,
    "furious": 16, "grateful": 17, "guilty": 18, "hopeful": 19,
    "impressed": 20, "jealous": 21, "joyful": 22, "lonely": 23,
    "nostalgic": 24, "prepared": 25, "proud": 26, "sad": 27,
    "sentimental": 28, "surprised": 29, "terrified": 30, "trusting": 31,
}

NUM_EMOTION_CLASSES_PHASE1 = 32   # EmpatheticDialogues
NUM_EMOTION_CLASSES_PHASE2 = 8    # ESConv coarse
NUM_STRATEGY_CLASSES = 8


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer training
# ─────────────────────────────────────────────────────────────────────────────

def train_tokenizer(
    corpus_files: List[str],
    save_dir: str,
    vocab_size: int = 10_000,
) -> Tokenizer:
    """
    Train a byte-level BPE tokenizer on the provided corpus files.

    IMPORTANT: All special tokens are added BEFORE BPE merges are computed
    so they are treated as atomic units and never split.

    Args:
        corpus_files: List of plain-text file paths (one sentence per line).
        save_dir:     Directory to save tokenizer artifacts.
        vocab_size:   Target vocabulary size (default 10,000 per spec).

    Returns:
        Trained tokenizer object.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Byte-level BPE: handles any Unicode without [UNK] at byte level
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # WHY special_tokens passed to trainer: ensures they appear in vocab
    # at their designated IDs and are never split by merge rules.
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=ALL_SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(f"Training BPE tokenizer on {len(corpus_files)} file(s) ...")
    tokenizer.train(files=corpus_files, trainer=trainer)

    # Add post-processor: automatically wrap sequences with [BOS] / [EOS]
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS]:0 $A:0 [EOS]:0",
        pair="[BOS]:0 $A:0 [SEP]:0 $B:0 [EOS]:0",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id),
                        ("[SEP]", tokenizer.token_to_id("[SEP]"))],
    )

    save_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path} (vocab_size={tokenizer.get_vocab_size()})")

    # Save human-readable ID → token mapping for debugging
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(os.path.join(save_dir, "vocab_readable.json"), "w") as f:
        json.dump({str(v): k for k, v in sorted_vocab}, f, indent=2)

    return tokenizer


def load_tokenizer(save_dir: str) -> Tokenizer:
    """Load a previously saved tokenizer from disk."""
    path = os.path.join(save_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No tokenizer found at {path}. Run train_tokenizer first.")
    tok = Tokenizer.from_file(path)
    print(f"Loaded tokenizer (vocab_size={tok.get_vocab_size()}) from {path}")
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class HappyBotTokenizer:
    """
    Thin wrapper around the HuggingFace fast tokenizer.

    Provides encode/decode helpers used throughout the training pipeline
    and exposes token IDs for all special tokens as properties.
    """

    def __init__(self, save_dir: str):
        self._tok = load_tokenizer(save_dir)
        self._tok.enable_padding(pad_id=self.pad_id, pad_token="[PAD]")

    # ── Special token IDs ─────────────────────────────────────────────────

    def _id(self, token: str) -> int:
        idx = self._tok.token_to_id(token)
        if idx is None:
            raise KeyError(f"Token '{token}' not in vocabulary.")
        return idx

    @property
    def pad_id(self) -> int:
        return self._id("[PAD]")

    @property
    def bos_id(self) -> int:
        return self._id("[BOS]")

    @property
    def eos_id(self) -> int:
        return self._id("[EOS]")

    @property
    def unk_id(self) -> int:
        return self._id("[UNK]")

    @property
    def cls_id(self) -> int:
        return self._id("[CLS]")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def strategy_token_id(self, strategy_key: str) -> int:
        """Return the vocab ID for a strategy control token."""
        return self._id(f"[strategy_{strategy_key}]")

    def emotion_token_id(self, emotion_label: str, source: str = "empathetic") -> int:
        """Return vocab ID for an emotion token (source: 'empathetic' | 'esconv')."""
        if source == "esconv":
            return self._id(f"[seeker_emotion_{emotion_label}]")
        return self._id(f"[emotion_{emotion_label}]")

    def intensity_token_id(self, intensity: int) -> int:
        return self._id(f"[intensity_{intensity}]")

    # ── Encode / Decode ───────────────────────────────────────────────────

    def encode(self, text: str, max_length: Optional[int] = 512) -> List[int]:
        """
        Encode a string to token IDs.
        The post-processor prepends [BOS] and appends [EOS] automatically.
        Truncates to max_length if needed.
        """
        enc = self._tok.encode(text)
        ids = enc.ids
        if max_length and len(ids) > max_length:
            # Truncate but keep [EOS] at end
            ids = ids[:max_length - 1] + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str], max_length: int = 512) -> List[List[int]]:
        """Batch encode without padding (collate_fn handles that)."""
        return [self.encode(t, max_length) for t in texts]
