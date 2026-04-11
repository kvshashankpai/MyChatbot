"""
train_phase1.py — Phase 1: Base Pre-Training on EmpatheticDialogues

Goal (Section 4.1):
  Teach the model basic English syntax, broad emotional vocabulary, and the
  fundamental emotion-to-response mapping before exposing it to clinical data.

Key settings:
  • Peak LR: 1e-4 with 500-step linear warmup
  • Cosine annealing after warmup
  • 15-25 epochs (until validation perplexity plateaus)
  • Strategy loss weight = 0.0 (EmpatheticDialogues has no strategy labels)
  • Emotion loss weight  = 0.3
  • Teacher forcing:     100% throughout
  • Effective batch size: 128 (accum_steps=4, micro_batch=32)
  • Gradient clipping:   1.0
  • Label smoothing:     0.1

FIX NOTES vs original:
  - Device: auto-detect MPS (Apple Silicon) / CUDA / CPU instead of hard-
    coding CPU. Training on CPU with 50K examples is impractically slow.
  - GradAccumulator.should_step() was being called BEFORE incrementing the
    internal counter in some implementations; replaced with a simple modulo
    counter that is transparent and correct.
  - Added --debug_nans flag: runs the first batch with debug_nans=True to
    print per-tensor stats. Use this to isolate NaN origin if it recurs.
  - NaN batch skip now also zeros gradients to prevent accumulating stale
    gradient from a partially-completed backward.
  - Added a sanity_check() function that does a single forward+backward on
    a tiny synthetic batch before the main loop, so misconfiguration is
    caught immediately with a clear error message.

ROOT-CAUSE FIX (NaN total loss at sanity check):
  The model tensors and individual losses (L_gen, L_emotion) were all finite,
  but L_total was NaN. The culprit was IEEE-754 arithmetic:

      0.0 * nan  ==  nan   (always)

  CrossEntropyLoss(ignore_index=-1) returns nan when ALL labels in the batch
  are -1 (i.e. every sample is "ignored"). In Phase 1, strategy_labels are
  always -1 (EmpatheticDialogues has no strategy annotations), so
  L_strategy = nan. Multiplying by lambda_strategy=0.0 does NOT zero it out:

      L_total = L_gen + 0.3*L_emotion + 0.0*nan  ==>  nan

  FIX applied inside compute_total_loss() (reproduced here inline):
    - Skip the weighted add entirely when lambda == 0.0 (guard with `if`).
    - Also guard when the loss tensor itself is non-finite (all-ignored batch).
  This is the ONLY change that was needed; the model architecture is correct.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import HappyBotTokenizer, NUM_EMOTION_CLASSES_PHASE1
from dataset import HappyBotDataset, build_dataloader
from model.transformer import HappyBot
from utils import (
    get_logger,
    get_warmup_cosine_scheduler,
    build_generation_loss,
    build_classification_loss,
    compute_perplexity,
    compute_distinct_n,
    compute_f1_score,
    build_optimizer,
    save_checkpoint,
    load_checkpoint,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Happy-Bot Phase 1 Training")
    p.add_argument("--data_dir",       default="data/processed")
    p.add_argument("--tokenizer_dir",  default="data/tokenizer")
    p.add_argument("--checkpoint_dir", default="checkpoints/phase1")
    p.add_argument("--log_dir",        default="logs")
    p.add_argument("--epochs",         type=int,   default=20)
    p.add_argument("--micro_batch",    type=int,   default=32)
    p.add_argument("--accum_steps",    type=int,   default=4,
                   help="Gradient accumulation steps (effective_batch = micro * accum)")
    p.add_argument("--peak_lr",        type=float, default=1e-4)
    p.add_argument("--warmup_steps",   type=int,   default=500)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--max_src_len",    type=int,   default=512)
    p.add_argument("--max_tgt_len",    type=int,   default=128)
    p.add_argument("--d_model",        type=int,   default=256)
    p.add_argument("--num_heads",      type=int,   default=2)
    p.add_argument("--num_layers",     type=int,   default=4)
    p.add_argument("--d_ff",           type=int,   default=1024)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--label_smooth",   type=float, default=0.1)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--log_every",      type=int,   default=100)
    p.add_argument("--eval_every",     type=int,   default=500)
    p.add_argument("--resume",         default=None)
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--debug_nans",     action="store_true",
                   help="Print per-tensor stats for the first batch (NaN hunting)")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Device selection
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# NaN-safe total loss
# ─────────────────────────────────────────────────────────────────────────────

def compute_total_loss(
    gen_logits,
    decoder_targets,
    emotion_logits,
    emotion_labels,
    strategy_logits,
    strategy_labels,
    gen_loss_fn,
    emotion_loss_fn,
    strategy_loss_fn,
    lambda_emotion: float = 0.3,
    lambda_strategy: float = 0.0,
):
    """
    Compute the weighted multi-task loss.

    BUG FIX — IEEE-754 NaN propagation:
      CrossEntropyLoss(ignore_index=-1) returns nan when every label in the
      batch equals -1 (all samples are "ignored"). In Phase 1, strategy_labels
      are always -1, so L_strategy = nan.

      The original code then computed:
          L_total = L_gen + lambda_emotion * L_emotion + lambda_strategy * L_strategy
                  = finite + finite + 0.0 * nan
                  = nan          ← because 0.0 * nan == nan in IEEE-754

      FIX: skip the weighted addition entirely when lambda == 0.0 or when the
      component loss is non-finite (all-ignored batch).  This is mathematically
      equivalent to zero contribution while avoiding NaN pollution.

    Args:
        gen_logits:       (B, T, V) — decoder output logits
        decoder_targets:  (B, T)    — token IDs (pad=0 is ignored by loss)
        emotion_logits:   (B, E)
        emotion_labels:   (B,)      — -1 means "no label / ignore"
        strategy_logits:  (B, 8)
        strategy_labels:  (B,)      — -1 means "no label / ignore"
        gen_loss_fn:      LabelSmoothingCrossEntropy (or nn.CrossEntropyLoss)
        emotion_loss_fn:  nn.CrossEntropyLoss(ignore_index=-1)
        strategy_loss_fn: nn.CrossEntropyLoss(ignore_index=-1)
        lambda_emotion:   Weight on emotion loss term.
        lambda_strategy:  Weight on strategy loss term (0.0 in Phase 1).

    Returns:
        L_total (scalar tensor), loss_dict (Python floats for logging)
    """
    # Generation loss — pass (B, T, V) and (B, T) directly.
    # build_generation_loss() returns a module whose forward() does its own
    # reshape internally (it unpacks B, T, V = logits.shape on line 144 of
    # utils.py). Reshaping here before calling it caused the
    # "not enough values to unpack (expected 3, got 2)" error.
    L_gen = gen_loss_fn(gen_logits, decoder_targets)

    L_total = L_gen  # start accumulator with the always-present generation loss

    # ── Emotion loss ───────────────────────────────────────────────────────
    # Guard: only add if lambda > 0 AND the loss is finite (all-ignore → nan).
    L_emotion_val = 0.0
    if lambda_emotion > 0.0:
        L_emotion = emotion_loss_fn(emotion_logits, emotion_labels)
        if torch.isfinite(L_emotion):
            L_total = L_total + lambda_emotion * L_emotion
            L_emotion_val = L_emotion.item()
        # else: entire mini-batch had ignore_index labels → skip silently

    # ── Strategy loss ──────────────────────────────────────────────────────
    # Guard: skip when lambda == 0.0 to avoid  0.0 * nan = nan  (IEEE-754).
    L_strategy_val = 0.0
    if lambda_strategy > 0.0:
        L_strategy = strategy_loss_fn(strategy_logits, strategy_labels)
        if torch.isfinite(L_strategy):
            L_total = L_total + lambda_strategy * L_strategy
            L_strategy_val = L_strategy.item()

    loss_dict = {
        "L_total":    L_total.item(),
        "L_gen":      L_gen.item(),
        "L_emotion":  L_emotion_val,
        "L_strategy": L_strategy_val,
    }

    return L_total, loss_dict


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check: one forward + backward on synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(model, gen_loss_fn, emotion_loss_fn, strategy_loss_fn,
                 vocab_size, pad_id, device, logger):
    """
    Run a single forward+backward on a tiny random batch.
    Catches init bugs (NaN weights, wrong shapes) before the real loop.
    Raises RuntimeError if anything is wrong.
    """
    logger.info("Running sanity check on synthetic batch …")
    model.train()
    B, S, T = 2, 20, 10

    enc_ids   = torch.randint(1, vocab_size, (B, S), device=device)
    dec_in    = torch.randint(1, vocab_size, (B, T), device=device)
    dec_tgt   = torch.randint(1, vocab_size, (B, T), device=device)
    enc_mask  = torch.zeros(B, S, dtype=torch.bool, device=device)
    dec_mask  = torch.zeros(B, T, dtype=torch.bool, device=device)
    emo_lbl   = torch.zeros(B, dtype=torch.long, device=device)
    # All strategy labels are -1 in Phase 1 — this is the case that previously
    # produced nan via 0.0 * CrossEntropyLoss(all-ignored) = 0.0 * nan = nan.
    strat_lbl = torch.full((B,), -1, dtype=torch.long, device=device)

    out = model(enc_ids, dec_in, enc_mask, dec_mask, debug_nans=True)

    L, loss_dict = compute_total_loss(
        gen_logits=out["gen_logits"],
        decoder_targets=dec_tgt,
        emotion_logits=out["emotion_logits"],
        emotion_labels=emo_lbl,
        strategy_logits=out["strategy_logits"],
        strategy_labels=strat_lbl,
        gen_loss_fn=gen_loss_fn,
        emotion_loss_fn=emotion_loss_fn,
        strategy_loss_fn=strategy_loss_fn,
        lambda_emotion=0.3,
        lambda_strategy=0.0,   # Phase 1: no strategy supervision
    )

    if not torch.isfinite(L):
        raise RuntimeError(
            f"Sanity check FAILED: loss={L.item():.6f}  dict={loss_dict}\n"
            "This means the model produces NaN/Inf on the very first forward pass.\n"
            "Re-run with --debug_nans to see which tensor is bad."
        )

    L.backward()

    # Check for NaN gradients
    nan_params = [
        name for name, p in model.named_parameters()
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if nan_params:
        raise RuntimeError(
            f"Sanity check FAILED: NaN/Inf gradients in {nan_params[:5]} …\n"
            "Check embedding scale, weight tying, or loss formulation."
        )

    model.zero_grad()
    logger.info(f"Sanity check PASSED — initial loss={L.item():.4f}  dict={loss_dict}")


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: HappyBot,
    val_loader: DataLoader,
    gen_loss_fn,
    emotion_loss_fn,
    device: torch.device,
    tokenizer: HappyBotTokenizer,
    max_batches: int = 50,
) -> dict:
    model.eval()

    total_loss     = 0.0
    total_gen_loss = 0.0
    emotion_preds  = []
    emotion_tgts   = []
    n_batches      = 0

    dummy_strategy_loss = nn.CrossEntropyLoss(ignore_index=-1)

    for batch in val_loader:
        if n_batches >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            encoder_ids=batch["encoder_ids"],
            decoder_input=batch["decoder_input"],
            encoder_mask=batch["encoder_mask"],
            decoder_mask=batch["decoder_mask"],
        )

        _, loss_dict = compute_total_loss(
            gen_logits=out["gen_logits"],
            decoder_targets=batch["decoder_target"],
            emotion_logits=out["emotion_logits"],
            emotion_labels=batch["emotion_label"],
            strategy_logits=out["strategy_logits"],
            strategy_labels=batch["strategy_label"],
            gen_loss_fn=gen_loss_fn,
            emotion_loss_fn=emotion_loss_fn,
            strategy_loss_fn=dummy_strategy_loss,
            lambda_emotion=0.3,
            lambda_strategy=0.0,
        )

        total_loss     += loss_dict["L_total"]
        total_gen_loss += loss_dict["L_gen"]

        preds = out["emotion_logits"].argmax(dim=-1).cpu().tolist()
        tgts  = batch["emotion_label"].cpu().tolist()
        emotion_preds.extend(preds)
        emotion_tgts.extend([t for t in tgts if t >= 0])

        n_batches += 1

    if n_batches == 0:
        return {"val_loss": float("inf"), "perplexity": float("inf"), "emotion_f1": 0.0}

    avg_gen_loss = total_gen_loss / n_batches
    perplexity   = compute_perplexity(avg_gen_loss)
    emotion_f1   = compute_f1_score(emotion_preds, emotion_tgts,
                                    num_classes=NUM_EMOTION_CLASSES_PHASE1)

    model.train()
    return {
        "val_loss":   total_loss / n_batches,
        "perplexity": perplexity,
        "emotion_f1": emotion_f1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    device = get_device()
    logger = get_logger("phase1", log_file=f"{args.log_dir}/phase1.log")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="happy-bot", name="phase1", config=vars(args))
        except ImportError:
            logger.warning("wandb not installed — skipping. pip install wandb")
            args.use_wandb = False

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = HappyBotTokenizer(args.tokenizer_dir)
    vocab_size = tokenizer.vocab_size
    pad_id     = tokenizer.pad_id
    logger.info(f"Tokenizer loaded: vocab_size={vocab_size}")

    # ── Datasets ───────────────────────────────────────────────────────────
    train_path = os.path.join(args.data_dir, "empathetic_train.jsonl")
    val_path   = os.path.join(args.data_dir, "empathetic_val.jsonl")

    train_ds = HappyBotDataset(train_path, tokenizer,
                                args.max_src_len, args.max_tgt_len, phase=1)
    val_ds   = HappyBotDataset(val_path,   tokenizer,
                                args.max_src_len, args.max_tgt_len, phase=1)

    # pin_memory is only valid for CUDA; disable for MPS/CPU
    pin = device.type == "cuda"
    train_loader = build_dataloader(train_ds, args.micro_batch, pad_id,
                                    shuffle=True,  pin_memory=pin)
    val_loader   = build_dataloader(val_ds,   args.micro_batch, pad_id,
                                    shuffle=False, pin_memory=pin)

    # ── Model ──────────────────────────────────────────────────────────────
    model = HappyBot(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_emotion_classes=NUM_EMOTION_CLASSES_PHASE1,
        num_strategy_classes=8,
        pad_token_id=pad_id,
    ).to(device)

    # ── Loss functions ──────────────────────────────────────────────────────
    gen_loss_fn      = build_generation_loss(vocab_size, pad_id, args.label_smooth)
    emotion_loss_fn  = build_classification_loss(class_weights=None)
    strategy_loss_fn = build_classification_loss(class_weights=None)

    # ── Sanity check ───────────────────────────────────────────────────────
    sanity_check(model, gen_loss_fn, emotion_loss_fn, strategy_loss_fn,
                 vocab_size, pad_id, device, logger)

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    optimizer   = build_optimizer(model, args.peak_lr, args.weight_decay)
    total_steps = (len(train_loader) // args.accum_steps) * args.epochs
    scheduler   = get_warmup_cosine_scheduler(optimizer, args.warmup_steps, total_steps)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step",  0)

    # ── Training loop ───────────────────────────────────────────────────────
    best_perplexity = float("inf")
    logger.info(f"Starting Phase 1 training: {args.epochs} epochs, "
                f"{total_steps} total optimiser steps")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad()

        epoch_start  = time.time()
        running_loss = {"L_total": 0.0, "L_gen": 0.0, "L_emotion": 0.0}
        micro_count  = 0   # counts micro-batches within the current accum window

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            if batch_idx == 0 and epoch == start_epoch:
                logger.info(f"Sample encoder_ids[:20]: {batch['encoder_ids'][0][:20].tolist()}")
                logger.info(f"Sample decoder_target[:20]: {batch['decoder_target'][0][:20].tolist()}")

            # ── Forward pass ───────────────────────────────────────────────
            debug = args.debug_nans and batch_idx == 0 and epoch == start_epoch
            out = model(
                encoder_ids=batch["encoder_ids"],
                decoder_input=batch["decoder_input"],
                encoder_mask=batch["encoder_mask"],
                decoder_mask=batch["decoder_mask"],
                debug_nans=debug,
            )

            # ── Loss ───────────────────────────────────────────────────────
            L_total, loss_dict = compute_total_loss(
                gen_logits=out["gen_logits"],
                decoder_targets=batch["decoder_target"],
                emotion_logits=out["emotion_logits"],
                emotion_labels=batch["emotion_label"],
                strategy_logits=out["strategy_logits"],
                strategy_labels=batch["strategy_label"],
                gen_loss_fn=gen_loss_fn,
                emotion_loss_fn=emotion_loss_fn,
                strategy_loss_fn=strategy_loss_fn,
                lambda_emotion=0.3,
                lambda_strategy=0.0,
            )

            if not torch.isfinite(L_total):
                logger.warning(
                    f"NaN/Inf loss at epoch={epoch} batch={batch_idx} — "
                    f"L_gen={loss_dict.get('L_gen', '?'):.4f}  "
                    f"L_emo={loss_dict.get('L_emotion', '?'):.4f}  "
                    "Skipping and zeroing gradients."
                )
                optimizer.zero_grad()
                micro_count = 0
                continue

            # Scale by accum_steps so effective gradient magnitude is correct
            (L_total / args.accum_steps).backward()
            micro_count += 1

            for k in running_loss:
                running_loss[k] += loss_dict.get(k, 0.0)

            # ── Optimizer step every accum_steps micro-batches ─────────────
            if micro_count % args.accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # ── Logging ───────────────────────────────────────────────
                if global_step % args.log_every == 0:
                    avg = {k: v / args.log_every for k, v in running_loss.items()}
                    lr  = scheduler.get_last_lr()[0]
                    logger.info(
                        f"E{epoch:03d} step={global_step:06d}  "
                        f"L={avg['L_total']:.4f}  "
                        f"L_gen={avg['L_gen']:.4f}  "
                        f"L_emo={avg['L_emotion']:.4f}  "
                        f"lr={lr:.2e}"
                    )
                    if args.use_wandb:
                        import wandb
                        wandb.log({"train/" + k: v for k, v in avg.items()},
                                  step=global_step)
                        wandb.log({"lr": lr}, step=global_step)
                    running_loss = {k: 0.0 for k in running_loss}

                # ── Validation ────────────────────────────────────────────
                if global_step % args.eval_every == 0:
                    val_metrics = evaluate(
                        model, val_loader, gen_loss_fn, emotion_loss_fn,
                        device, tokenizer
                    )
                    logger.info(
                        f"  [VAL] perplexity={val_metrics['perplexity']:.2f}  "
                        f"emotion_f1={val_metrics['emotion_f1']:.4f}  "
                        f"val_loss={val_metrics['val_loss']:.4f}"
                    )
                    if args.use_wandb:
                        import wandb
                        wandb.log({"val/" + k: v for k, v in val_metrics.items()},
                                  step=global_step)

                    if val_metrics["perplexity"] < best_perplexity:
                        best_perplexity = val_metrics["perplexity"]
                        save_checkpoint(
                            model, optimizer, scheduler,
                            epoch, global_step, best_perplexity,
                            metric_name="perplexity",
                            checkpoint_dir=args.checkpoint_dir,
                            filename="best_model.pt",
                        )

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} complete in {epoch_time:.1f}s")

    # Final checkpoint
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, global_step,
                    best_perplexity, "perplexity", args.checkpoint_dir,
                    filename="final_model.pt")

    logger.info(f"Phase 1 training complete. Best perplexity: {best_perplexity:.2f}")
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train(parse_args())