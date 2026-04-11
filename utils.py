"""
utils.py — Training utilities: LR scheduler, loss functions, metrics, checkpointing.
"""

import math
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


# ── Learning Rate Scheduler ───────────────────────────────────────────────────

class WarmupCosineScheduler:
    """
    Linear warmup followed by cosine annealing.
    This is non-negotiable for stable transformer training from random weights.
    
    Usage:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=500, total_steps=10000, peak_lr=1e-4)
        for step in training_loop:
            scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        peak_lr: float,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self) -> float:
        step = self._step
        if step <= self.warmup_steps:
            # Linear warmup from 0 to peak_lr
            return self.peak_lr * step / max(1, self.warmup_steps)
        else:
            # Cosine annealing from peak_lr to min_lr
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine_factor

    def state_dict(self) -> dict:
        return {"step": self._step, "peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state: dict):
        self._step = state["step"]


# ── Loss Functions ────────────────────────────────────────────────────────────

class LabelSmoothedCrossEntropy(nn.Module):
    """
    Label-smoothed cross-entropy loss for generation.
    smoothing=0.1: distributes 0.1 probability mass uniformly over all vocab tokens.
    This prevents overconfident predictions and improves output diversity.
    ignore_index=-100: ignores padded positions.
    """

    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T, V) — raw (un-softmaxed) logits
        targets: (B, T) — ground-truth token ids, padded positions = ignore_index
        """
        B, T, V = logits.size()
        logits_flat = logits.reshape(-1, V)   # (B*T, V)
        targets_flat = targets.reshape(-1)    # (B*T,)

        # Mask padding positions
        valid = targets_flat != self.ignore_index
        if valid.sum() == 0:
            return logits.new_zeros(())

        logits_flat = logits_flat[valid]
        targets_flat = targets_flat[valid]

        # Log softmax
        log_probs = F.log_softmax(logits_flat, dim=-1)

        # One-hot smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (V - 1))
            smooth_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for HappyBot training.
    L_total = L_gen + lambda_emotion * L_emotion + lambda_strategy * L_strategy
    
    During Phase 1 (EmpatheticDialogues):
      - lambda_strategy = 0.0 (no strategy annotations available)
      - strategy labels should be passed as -1 (masked out)
    
    During Phase 2 (ESConv fine-tuning):
      - Both lambda values = 0.3
      - strategy_weights: class-frequency-inverse weights for imbalance correction
    """

    def __init__(
        self,
        vocab_size: int,
        num_emotion_classes: int = 32,
        num_strategy_classes: int = 8,
        label_smoothing: float = 0.1,
        lambda_emotion: float = 0.3,
        lambda_strategy: float = 0.3,
        strategy_weights: torch.Tensor = None,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.lambda_emotion = lambda_emotion
        self.lambda_strategy = lambda_strategy

        self.gen_loss_fn = LabelSmoothedCrossEntropy(
            vocab_size=vocab_size,
            smoothing=label_smoothing,
            ignore_index=-100,
        )
        self.emotion_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.strategy_loss_fn = nn.CrossEntropyLoss(
            weight=strategy_weights,
            ignore_index=-1,
        )

    def forward(
        self,
        logits: torch.Tensor,            # (B, T, vocab_size) — generation logits
        tgt_labels: torch.Tensor,        # (B, T) — shifted decoder targets
        emotion_logits: torch.Tensor,    # (B, num_emotion_classes)
        emotion_labels: torch.Tensor,    # (B,)  — -1 = ignore
        strategy_logits: torch.Tensor,   # (B, num_strategy_classes)
        strategy_labels: torch.Tensor,   # (B,)  — -1 = ignore (Phase 1)
    ) -> dict:
        L_gen = self.gen_loss_fn(logits, tgt_labels)

        # Emotion loss (active in both phases)
        L_emotion = self.emotion_loss_fn(emotion_logits, emotion_labels)

        # Strategy loss (masked in Phase 1 via ignore_index=-1)
        if self.lambda_strategy > 0:
            L_strategy = self.strategy_loss_fn(strategy_logits, strategy_labels)
        else:
            L_strategy = logits.new_zeros(())

        L_total = L_gen + self.lambda_emotion * L_emotion + self.lambda_strategy * L_strategy

        return {
            "total": L_total,
            "generation": L_gen,
            "emotion": L_emotion,
            "strategy": L_strategy,
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_distinct_n(token_sequences: list[list[int]], n: int) -> float:
    """
    Distinct-n: ratio of unique n-grams to total n-grams across all generated sequences.
    Measures response diversity. Distinct-1 > 0.15, Distinct-2 > 0.40 are targets.
    """
    all_ngrams = []
    for seq in token_sequences:
        for i in range(len(seq) - n + 1):
            all_ngrams.append(tuple(seq[i:i + n]))
    if not all_ngrams:
        return 0.0
    unique = len(set(all_ngrams))
    total = len(all_ngrams)
    return unique / total


def compute_perplexity(loss: float) -> float:
    """Perplexity from cross-entropy loss (nats)."""
    return math.exp(min(loss, 100))  # clamp to avoid overflow


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1) -> float:
    """Classification accuracy for NLU heads."""
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).float().sum()
    return (correct / mask.sum()).item()


# ── Strategy Class Weights ────────────────────────────────────────────────────

def compute_strategy_weights(
    strategy_counts_path: str,
    num_classes: int = 8,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for the ESConv strategy head.
    Loads the strategy_counts.json file produced by prepare_esconv.py.
    
    Weight formula: w_i = total / (num_classes * count_i)
    This is the standard sklearn balanced class weight formula.
    """
    if not os.path.exists(strategy_counts_path):
        # Return uniform weights if file not found
        return torch.ones(num_classes)

    with open(strategy_counts_path) as f:
        counts = json.load(f)  # {strategy_id_str: count}

    total = sum(counts.values())
    weights = torch.ones(num_classes)
    for sid_str, count in counts.items():
        sid = int(sid_str)
        if count > 0 and sid < num_classes:
            weights[sid] = total / (num_classes * count)

    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    if device:
        weights = weights.to(device)
    return weights


# ── AdamW Configuration ───────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.01) -> torch.optim.AdamW:
    """
    Build AdamW optimizer with weight decay applied correctly:
    - Exclude biases and LayerNorm parameters from weight decay
    - This matches the BERT/GPT training recipe for transformers
    """
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower() or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.98), eps=1e-9)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    metrics: dict,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }, path)
    print(f"  [ckpt] Saved checkpoint → {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"  [ckpt] Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt