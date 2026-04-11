"""
utils.py — Shared Utilities

Covers:
  • Learning rate scheduler (warmup + cosine annealing, Sections 4.1/4.2)
  • Label-smoothed cross-entropy loss (Section 4.4)
  • Class-weighted cross-entropy for strategy head (Section 2.5)
  • Distinct-1 / Distinct-2 diversity metrics (Section 8.2)
  • Perplexity computation
  • Checkpoint save/load
  • Logging helpers
"""

import os
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler: Linear warmup + cosine annealing (Sections 4.1/4.2)
# ─────────────────────────────────────────────────────────────────────────────

def get_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.05,
) -> LambdaLR:
    """
    Learning rate schedule:
      Phase A (0 → warmup_steps):   linear increase from 0 → peak LR
      Phase B (warmup_steps → end): cosine annealing from peak → min_lr

    WHY linear warmup (critical, per spec):
      At random initialisation the attention weights are unstable.
      Starting with a large LR causes very large gradient steps that
      corrupt the attention matrices before the model has a chance to
      form meaningful representations. Warmup gives the optimizer time
      to stabilise.

    WHY cosine annealing:
      Smooth decay avoids the abrupt LR drops of step schedules.
      The cosine curve naturally slows near convergence, allowing
      the model to settle into a local minimum without bouncing.

    Args:
        optimizer:    AdamW or Adam optimizer.
        warmup_steps: Number of linear warmup steps (500 Phase1, 100 Phase2).
        total_steps:  Total training steps.
        min_lr_ratio: Minimum LR = peak_LR * min_lr_ratio.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear increase: 0 → 1.0 over warmup_steps
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing: 1.0 → min_lr_ratio
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothedCrossEntropyLoss(nn.Module):
    """
    Label-smoothed cross-entropy for the generation head (Section 4.4).

    WHY label smoothing (epsilon=0.1):
      Without smoothing the model assigns 100% probability to a single token,
      which leads to overconfident predictions and reduced diversity in the
      generated responses (low Distinct-2 score).
      Smoothing distributes a fraction (epsilon) of the probability mass
      uniformly across all vocabulary tokens, acting as a soft regularizer.

    Formula:
      loss = (1 - eps) * NLL_loss + eps * uniform_distribution_loss
           = (1 - eps) * NLL_loss + eps * mean(-log(1/V))

    Implementation avoids F.one_hot to stay memory-efficient with large vocab.

    Args:
        vocab_size:   Size of vocabulary.
        padding_idx:  Token ID to ignore (PAD positions, typically 0).
        smoothing:    Smoothing factor epsilon (0.1 per spec).
    """

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size  = vocab_size
        self.padding_idx = padding_idx
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(
        self,
        logits: torch.Tensor,   # (B, T, V) raw logits from decoder
        targets: torch.Tensor,  # (B, T)   target token IDs; -100 at PAD positions
    ) -> torch.Tensor:
        """
        Returns:
            Scalar mean loss (averaged over non-padding positions).
        """
        B, T, V = logits.shape

        # Flatten to (B*T, V) and (B*T,) for vectorised computation
        logits_flat  = logits.reshape(-1, V)   # (B*T, V)
        targets_flat = targets.reshape(-1)     # (B*T,)

        # Log-softmax for numerical stability (equivalent to log(softmax(x)))
        log_probs = F.log_softmax(logits_flat, dim=-1)  # (B*T, V)

        # True NLL loss for the correct token
        nll_loss = F.nll_loss(
            log_probs,
            targets_flat,
            ignore_index=-100,   # -100 marks PAD in collate_fn
            reduction="sum",
        )

        # Smoothed loss: average log-prob across all vocab tokens
        # We mask out PAD positions (targets == -100) manually
        pad_mask = (targets_flat == -100)                        # (B*T,)
        smooth_loss = -log_probs.sum(dim=-1)                     # (B*T,) per-position sum
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.sum()

        # Count non-padding tokens for normalisation
        n_tokens = (~pad_mask).sum().float()

        total_loss = (
            self.confidence * nll_loss
            + self.smoothing * smooth_loss / V
        ) / (n_tokens + 1e-9)

        return total_loss


def build_generation_loss(vocab_size: int, pad_id: int = 0, smoothing: float = 0.1):
    """Factory for the generation loss (label-smoothed CE)."""
    return LabelSmoothedCrossEntropyLoss(vocab_size, pad_id, smoothing)


def build_classification_loss(
    class_weights: Optional[torch.Tensor] = None,
) -> nn.CrossEntropyLoss:
    """
    Cross-entropy loss for emotion and strategy heads.

    For strategy (Phase 2): class_weights tensor handles imbalance.
    For emotion: no weighting (32 EmpatheticDialogues classes are reasonably balanced).
    Targets with value -1 (EmpatheticDialogues has no strategy label) are ignored.
    """
    return nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Task Loss Computation (Section 4.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_loss(
    gen_logits: torch.Tensor,           # (B, T, V)
    decoder_targets: torch.Tensor,      # (B, T)
    emotion_logits: torch.Tensor,       # (B, num_emotion_classes)
    emotion_labels: torch.Tensor,       # (B,)
    strategy_logits: torch.Tensor,      # (B, 8)
    strategy_labels: torch.Tensor,      # (B,)
    gen_loss_fn: LabelSmoothedCrossEntropyLoss,
    emotion_loss_fn: nn.CrossEntropyLoss,
    strategy_loss_fn: nn.CrossEntropyLoss,
    lambda_emotion: float = 0.3,
    lambda_strategy: float = 0.3,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the unified multi-task loss:
        L_total = L_gen + lambda_1 * L_emotion + lambda_2 * L_strategy

    WHY one backward() on L_total:
      Gradients flow simultaneously through decoder (L_gen), encoder NLU heads
      (L_emotion, L_strategy), AND back into the encoder's self-attention layers
      via the heads. This forces the encoder to develop psychologically
      meaningful representations, which then improves cross-attention quality.

    Returns:
        (L_total, dict of individual loss values for logging)
    """
    # Generation loss (decoder output vs target tokens)
    L_gen = gen_loss_fn(gen_logits, decoder_targets)

    # Emotion classification loss
    L_emotion = emotion_loss_fn(emotion_logits, emotion_labels)

    # Strategy classification loss (class-weighted in Phase 2)
    L_strategy = strategy_loss_fn(strategy_logits, strategy_labels)

    # Unified loss (lambda weights per spec)
    L_total = L_gen + lambda_emotion * L_emotion + lambda_strategy * L_strategy

    # Return individual losses for monitoring (detect which head is struggling)
    loss_dict = {
        "L_total":    L_total.item(),
        "L_gen":      L_gen.item(),
        "L_emotion":  L_emotion.item() if not torch.isnan(L_emotion) else 0.0,
        "L_strategy": L_strategy.item() if not torch.isnan(L_strategy) else 0.0,
    }

    return L_total, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(avg_nll_loss: float) -> float:
    """Perplexity = exp(average NLL loss). Target < 80 per spec."""
    return math.exp(min(avg_nll_loss, 100))  # cap to avoid overflow


def compute_distinct_n(
    generated_texts: List[str],
    n: int,
) -> float:
    """
    Distinct-N: ratio of unique n-grams to total n-grams across all responses.

    Distinct-1 target: > 0.15 (word-level diversity)
    Distinct-2 target: > 0.40 (bigram-level diversity)

    WHY Distinct-2 is the primary early-stopping criterion (Section 4.4):
      A model can achieve low perplexity by generating fluent but repetitive
      text (e.g., always outputting "I understand how you feel.").
      Distinct-2 measures whether the model produces VARIED responses across
      different conversations, which is essential for therapeutic effectiveness.
    """
    all_ngrams   = []
    unique_ngrams = set()

    for text in generated_texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
        unique_ngrams.update(ngrams)

    if not all_ngrams:
        return 0.0

    return len(unique_ngrams) / len(all_ngrams)


def compute_f1_score(
    predictions: List[int],
    targets: List[int],
    num_classes: int,
    average: str = "weighted",
) -> float:
    """
    Weighted F1 score for emotion/strategy classification heads.

    WHY weighted F1 (not accuracy):
      Class imbalance means accuracy is misleading.
      Weighted F1 accounts for both precision and recall, weighted by
      class frequency — appropriate for the imbalanced strategy distribution.
    """
    try:
        from sklearn.metrics import f1_score
        return f1_score(targets, predictions, average=average,
                        labels=list(range(num_classes)), zero_division=0)
    except ImportError:
        # Fallback: macro F1 computed manually
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        support = [0] * num_classes

        for pred, tgt in zip(predictions, targets):
            if tgt < 0:
                continue
            support[tgt] += 1
            if pred == tgt:
                tp[tgt] += 1
            else:
                fp[pred] += 1
                fn[tgt]  += 1

        total = sum(support)
        weighted_f1 = 0.0
        for c in range(num_classes):
            prec = tp[c] / (tp[c] + fp[c] + 1e-9)
            rec  = tp[c] / (tp[c] + fn[c] + 1e-9)
            f1_c = 2 * prec * rec / (prec + rec + 1e-9)
            weighted_f1 += f1_c * (support[c] / (total + 1e-9))
        return weighted_f1


# ─────────────────────────────────────────────────────────────────────────────
# Teacher Forcing Ratio Scheduler (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def get_teacher_forcing_ratio(
    epoch: int,
    start_epoch: int = 0,
    total_epochs: int = 50,
    start_ratio: float = 1.0,
    end_ratio: float = 0.5,
) -> float:
    """
    Linear decay of teacher forcing ratio from start_ratio to end_ratio.

    WHY scheduled teacher forcing (Phase 2 only):
      During Phase 1 (random init), 100% teacher forcing is needed for
      stable training. In Phase 2, gradually exposing the model to its
      own prediction errors (scheduled sampling) reduces the exposure bias
      gap between training and inference, where no ground truth is available.

    Returns:
        Teacher forcing ratio for the current epoch (0.5 to 1.0).
    """
    if total_epochs <= 1:
        return start_ratio
    progress = (epoch - start_epoch) / (total_epochs - 1)
    ratio = start_ratio - progress * (start_ratio - end_ratio)
    return max(end_ratio, min(start_ratio, ratio))


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer builder (AdamW with selective weight decay)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    peak_lr: float = 1e-4,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """
    Build AdamW optimizer with weight decay applied only to weights (not biases
    or LayerNorm parameters).

    WHY exclude biases and LayerNorm from weight decay:
      Weight decay on biases has negligible regularization benefit but can
      interfere with the model's ability to learn necessary offsets.
      LayerNorm parameters (gamma and beta) are sensitive scale/shift values
      that should be unconstrained.
    """
    decay_params     = []
    no_decay_params  = []
    no_decay_names   = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name.lower():
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    print(f"AdamW: {len(decay_params)} decayed params, "
          f"{len(no_decay_params)} non-decayed params "
          f"({', '.join(no_decay_names[:5])}{'...' if len(no_decay_names) > 5 else ''})")

    return torch.optim.AdamW(param_groups, lr=peak_lr, betas=(0.9, 0.98), eps=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    metric_value: float,
    metric_name: str,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt",
) -> None:
    """Save full training checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        "epoch":          epoch,
        "step":           step,
        "model_state":    model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        f"best_{metric_name}": metric_value,
    }, path)
    print(f"Checkpoint saved: {path}  ({metric_name}={metric_value:.4f})")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    """Load checkpoint. Returns checkpoint dict for metadata access."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    print(f"Loaded checkpoint from {path} (epoch={ckpt.get('epoch', '?')})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Gradient accumulation helper
# ─────────────────────────────────────────────────────────────────────────────

class GradAccumulator:
    """
    Simulates a larger effective batch size via gradient accumulation.

    WHY: With batch_size=32 and accumulation_steps=4, the effective batch
    is 128 — matching the spec's "effective batch size = 128" requirement —
    without requiring 4× GPU memory.

    Usage:
        accum = GradAccumulator(steps=4)
        for batch in dataloader:
            loss = model(batch) / accum.steps   # scale loss
            loss.backward()
            if accum.should_step():
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
    """

    def __init__(self, steps: int = 4):
        self.steps = steps
        self._count = 0

    def should_step(self) -> bool:
        self._count += 1
        if self._count >= self.steps:
            self._count = 0
            return True
        return False

    def reset(self):
        self._count = 0
