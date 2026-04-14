"""
utils.py — Training utilities for HappyBot.

FIXES vs original:
  1. Added build_optimizer_with_layer_decay() — encoder params get a lower LR
     (peak_lr * encoder_lr_decay) than decoder params during Phase 2.
     This is the standard technique to prevent catastrophic forgetting when
     fine-tuning on a small dataset (ESConv ~1.6k samples).

  2. compute_strategy_weights() now handles the case where the counts dict
     has all 8 keys (as produced by the fixed prepare_esconv.py), and also
     guards against zero-count classes to avoid infinite weights.

  3. save_checkpoint() unchanged — stores all config keys inference.py needs.
  4. load_checkpoint() unchanged — returns raw dict, no model arg.
"""

import math
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


# ── LR Scheduler ─────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup → cosine annealing. Non-negotiable for from-scratch training."""

    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr, min_lr=1e-6):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.peak_lr      = peak_lr
        self.min_lr       = min_lr
        self._step        = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            # Scale the LR by each group's own lr_scale factor (set during optim construction)
            pg["lr"] = lr * pg.get("lr_scale", 1.0)
        return lr

    def _get_lr(self):
        s = self._step
        if s <= self.warmup_steps:
            return self.peak_lr * s / max(1, self.warmup_steps)
        prog = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        cos  = 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
        return self.min_lr + (self.peak_lr - self.min_lr) * cos

    def state_dict(self):
        return {"step": self._step, "peak_lr": self.peak_lr,
                "warmup_steps": self.warmup_steps}

    def load_state_dict(self, d):
        self._step = d["step"]


# ── Loss Functions ────────────────────────────────────────────────────────────

class LabelSmoothedCrossEntropy(nn.Module):
    """
    Label-smoothed CE for the generation head.
    ignore_index=-100 skips padded positions.
    """

    def __init__(self, vocab_size, smoothing=0.1, ignore_index=-100):
        super().__init__()
        self.vocab_size   = vocab_size
        self.smoothing    = smoothing
        self.confidence   = 1.0 - smoothing
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits:  (B, T, V)
        # targets: (B, T) — padded positions = ignore_index
        B, T, V = logits.shape
        lf = logits.reshape(-1, V)
        tf = targets.reshape(-1)

        valid = tf != self.ignore_index
        if not valid.any():
            return logits.new_zeros(())

        lf = lf[valid]
        tf = tf[valid]

        log_p = F.log_softmax(lf, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_p, self.smoothing / (V - 1))
            smooth.scatter_(1, tf.unsqueeze(1), self.confidence)
        return -(smooth * log_p).sum(dim=-1).mean()


class MultiTaskLoss(nn.Module):
    """
    L_total = L_gen + lambda_emotion * L_emotion + lambda_strategy * L_strategy

    Phase 1 (EmpatheticDialogues):  lambda_strategy = 0.0
    Phase 2 (ESConv):               lambda_strategy = 0.3

    Both classification heads use ignore_index=-1 so samples with label=-1
    contribute zero loss (handles missing annotations cleanly).
    """

    def __init__(
        self,
        vocab_size,
        num_emotion_classes=32,
        num_strategy_classes=8,
        label_smoothing=0.1,
        lambda_emotion=0.3,
        lambda_strategy=0.3,    # SET TO 0.0 FOR PHASE 1
        strategy_weights=None,
        ignore_index=-100,
    ):
        super().__init__()
        self.lambda_emotion  = lambda_emotion
        self.lambda_strategy = lambda_strategy

        self.gen_loss      = LabelSmoothedCrossEntropy(vocab_size, label_smoothing, ignore_index)
        self.emotion_loss  = nn.CrossEntropyLoss(ignore_index=-1)
        self.strategy_loss = nn.CrossEntropyLoss(weight=strategy_weights, ignore_index=-1)

    def forward(self, logits, tgt_labels, emotion_logits, emotion_labels,
                strategy_logits, strategy_labels):
        L_gen      = self.gen_loss(logits, tgt_labels)
        L_emotion  = self.emotion_loss(emotion_logits, emotion_labels)
        L_strategy = (
            self.strategy_loss(strategy_logits, strategy_labels)
            if self.lambda_strategy > 0 else logits.new_zeros(())
        )

        total = L_gen + self.lambda_emotion * L_emotion + self.lambda_strategy * L_strategy
        return {
            "total":      total,
            "generation": L_gen,
            "emotion":    L_emotion,
            "strategy":   L_strategy,
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_perplexity(loss: float) -> float:
    """exp(cross-entropy loss). Clamped to avoid overflow."""
    return math.exp(min(loss, 100.0))

# Alias kept for backward compatibility
calculate_perplexity = compute_perplexity


def compute_distinct_n(token_sequences, n: int) -> float:
    """
    Distinct-n: fraction of unique n-grams across all generated sequences.
    D1 target >0.15, D2 target >0.40 for non-collapsed generation.
    """
    all_ngrams = []
    for seq in token_sequences:
        for i in range(len(seq) - n + 1):
            all_ngrams.append(tuple(seq[i:i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_accuracy(logits, labels, ignore_index=-1):
    mask  = labels != ignore_index
    if not mask.any():
        return 0.0
    preds = logits.argmax(-1)
    return (preds[mask] == labels[mask]).float().mean().item()


# ── Strategy class weights ────────────────────────────────────────────────────

def compute_strategy_weights(counts_path, num_classes=8, device=None):
    """
    Balanced inverse-frequency weights from strategy_counts.json.
    FIX: guards against zero-count classes (would produce inf weight).
         Falls back to uniform if any class has 0 count — prevents NaN loss.
    """
    if not os.path.exists(counts_path):
        print(f"  [Warning] strategy_counts.json not found at {counts_path}. Using uniform weights.")
        w = torch.ones(num_classes)
        return w.to(device) if device else w

    with open(counts_path) as f:
        counts = json.load(f)

    # FIX: use smoothed counts — add 1 to every class to avoid div-by-zero
    total = sum(counts.values()) + num_classes  # adjusted for smoothing
    w = torch.ones(num_classes)
    for i in range(num_classes):
        cnt = counts.get(str(i), 0) + 1   # +1 Laplace smoothing
        w[i] = total / (num_classes * cnt)

    w = w / w.mean()   # normalise: mean weight = 1.0
    print(f"  [Strategy weights] min={w.min():.3f}  max={w.max():.3f}  "
          f"counts={[counts.get(str(i), 0) for i in range(num_classes)]}")
    return w.to(device) if device else w

# Alias kept for backward compatibility
calculate_strategy_weights = compute_strategy_weights


# ── Optimizers ────────────────────────────────────────────────────────────────

def build_optimizer(model, lr, weight_decay=0.01):
    """
    AdamW: weight decay on weights only, not biases or LayerNorm params.
    Used in Phase 1.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("bias", "norm", "embedding")):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.98), eps=1e-9,
    )


def build_optimizer_with_layer_decay(model, lr, encoder_lr_decay=0.5, weight_decay=0.01):
    """
    FIX: Layer-wise LR for Phase 2.

    Encoder params get lr * encoder_lr_decay (default 0.5×) so the
    pre-trained language modelling representations are updated slowly,
    preventing catastrophic forgetting on the small ESConv dataset.

    Decoder and classification head params get the full lr.

    The WarmupCosineScheduler reads lr_scale per param group and multiplies
    the scheduled LR by it, so the ratio is maintained throughout training.
    """
    # Separate encoder vs decoder/heads, and within each split by decay/no_decay
    groups = {
        "enc_decay":    [],
        "enc_no_decay": [],
        "dec_decay":    [],
        "dec_no_decay": [],
    }

    no_decay_keys = ("bias", "norm", "embedding")

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_encoder = name.startswith("encoder.")
        is_no_decay = any(k in name for k in no_decay_keys)

        if is_encoder and is_no_decay:
            groups["enc_no_decay"].append(p)
        elif is_encoder:
            groups["enc_decay"].append(p)
        elif is_no_decay:
            groups["dec_no_decay"].append(p)
        else:
            groups["dec_decay"].append(p)

    param_groups = [
        {"params": groups["enc_decay"],    "weight_decay": weight_decay, "lr_scale": encoder_lr_decay},
        {"params": groups["enc_no_decay"], "weight_decay": 0.0,          "lr_scale": encoder_lr_decay},
        {"params": groups["dec_decay"],    "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": groups["dec_no_decay"], "weight_decay": 0.0,          "lr_scale": 1.0},
    ]

    # Set initial lr per group
    for g in param_groups:
        g["lr"] = lr * g["lr_scale"]

    print(f"  [Optimizer] encoder_lr={lr * encoder_lr_decay:.2e}  decoder_lr={lr:.2e}")
    return torch.optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-9)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics, path):
    """
    Saves everything inference.py needs to reconstruct the model.
    Model config keys are read directly from the model object.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        # weights
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        # bookkeeping
        "epoch":   epoch,
        "step":    step,
        "metrics": metrics,
        # model architecture config (required by inference.py)
        "vocab_size":           model.vocab_size,
        "d_model":              model.d_model,
        "d_ff":                 model.d_ff,
        "num_heads":            model.num_heads,
        "num_encoder_layers":   model.num_encoder_layers,
        "num_decoder_layers":   model.num_decoder_layers,
        "dropout":              model.dropout_p,
        "pad_token_id":         model.pad_token_id,
    }, path)
    print(f"  [ckpt] Saved → {path}")


def load_checkpoint(path, map_location="cpu"):
    """
    Returns the raw checkpoint dict.
    Caller is responsible for model.load_state_dict(ckpt['model_state_dict']).
    """
    ckpt = torch.load(path, map_location=map_location)
    print(f"  [ckpt] Loaded from {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt