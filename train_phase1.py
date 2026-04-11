"""
train_phase1.py — Phase 1: pre-train HappyBot on EmpatheticDialogues.

Goal: Teach the model basic language, empathy patterns, and emotion representations.
Strategy head loss is MASKED (lambda=0) since EmpatheticDialogues has no strategy labels.
Emotion head trained with full weight.

Key training decisions enforced here:
  - Xavier initialization (handled in model modules)
  - AdamW with correct weight decay exclusion
  - Linear warmup (500 steps) + cosine annealing
  - Gradient clipping to 1.0 (prevents exploding gradients in 4-layer backprop)
  - Early stopping on validation perplexity
  - Checkpoint when val perplexity improves
"""

import argparse
import os
import math
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.transformer import HappyBot
from dataset import HappyBotDataset, collate_fn
from utils import (
    MultiTaskLoss,
    WarmupCosineScheduler,
    build_optimizer,
    compute_perplexity,
    save_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       default="data/processed")
    p.add_argument("--tokenizer_dir",  default="data/tokenizer")
    p.add_argument("--checkpoint_dir", default="checkpoints/phase1")
    p.add_argument("--epochs",         type=int, default=20)
    p.add_argument("--batch_size",     type=int, default=32)
    p.add_argument("--grad_accum",     type=int, default=4,    help="Effective batch = batch_size * grad_accum")
    p.add_argument("--peak_lr",        type=float, default=1e-4)
    p.add_argument("--warmup_steps",   type=int, default=500)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--max_grad_norm",  type=float, default=1.0)
    p.add_argument("--d_model",        type=int, default=256)
    p.add_argument("--num_heads",      type=int, default=2)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--d_ff",           type=int, default=1024)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--label_smoothing",type=float, default=0.1)
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--log_every",      type=int, default=100)
    p.add_argument("--patience",       type=int, default=5, help="Early stopping patience in epochs")
    return p.parse_args()


def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_gen = total_emo = total_batches = 0
    with torch.no_grad():
        for batch in loader:
            src        = batch["encoder_ids"].to(device)
            tgt_in     = batch["decoder_input_ids"].to(device)
            tgt_labels = batch["decoder_target_ids"].to(device)
            emo_labels = batch["emotion_label"].to(device)
            # Phase 1: no strategy labels
            strat_labels = torch.full((src.size(0),), -1, dtype=torch.long, device=device)

            out = model(src, tgt_in)
            losses = loss_fn(
                out["logits"], tgt_labels,
                out["emotion_logits"], emo_labels,
                out["strategy_logits"], strat_labels,
            )
            total_loss   += losses["total"].item()
            total_gen    += losses["generation"].item()
            total_emo    += losses["emotion"].item()
            total_batches += 1

    avg_gen = total_gen / max(1, total_batches)
    return {
        "total": total_loss / max(1, total_batches),
        "generation": avg_gen,
        "emotion": total_emo / max(1, total_batches),
        "perplexity": compute_perplexity(avg_gen),
    }


def main():
    args = parse_args()

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 1] Device: {device}")

    # ── Tokenizer ───────────────────────────────────────────────────────
    tokenizer_path = os.path.join(args.tokenizer_dir, "tokenizer.json")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    print(f"[Phase 1] Vocab size: {vocab_size}, PAD id: {pad_id}")

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = HappyBotDataset(
        os.path.join(args.data_dir, "empathetic_train.jsonl"),
        tokenizer, phase=1,
    )
    val_ds = HappyBotDataset(
        os.path.join(args.data_dir, "empathetic_val.jsonl"),
        tokenizer, phase=1,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id), num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id), num_workers=2,
    )
    print(f"[Phase 1] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    # ── Model ───────────────────────────────────────────────────────────
    model = HappyBot(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_emotion_classes=32,  # EmpatheticDialogues: 32 emotion labels
        num_strategy_classes=8,
        pad_token_id=pad_id,
    ).to(device)
    print(f"[Phase 1] Model parameters: {model.count_parameters():,}")

    # ── Loss ────────────────────────────────────────────────────────────
    # Phase 1: lambda_strategy = 0.0 (no strategy labels)
    loss_fn = MultiTaskLoss(
        vocab_size=vocab_size,
        num_emotion_classes=32,
        num_strategy_classes=8,
        label_smoothing=args.label_smoothing,
        lambda_emotion=0.3,
        lambda_strategy=0.0,   # Masked in Phase 1
        pad_token_id=pad_id,
    )

    # ── Optimizer + Scheduler ───────────────────────────────────────────
    optimizer = build_optimizer(model, lr=args.peak_lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps,
        total_steps=total_steps, peak_lr=args.peak_lr,
    )

    # ── W&B ─────────────────────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        wandb.init(project="happybot", name="phase1", config=vars(args))

    # ── Training Loop ───────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_ppl = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = epoch_gen = 0.0
        accum_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for step_i, batch in enumerate(pbar):
            src        = batch["encoder_ids"].to(device)
            tgt_in     = batch["decoder_input_ids"].to(device)
            tgt_labels = batch["decoder_target_ids"].to(device)
            emo_labels = batch["emotion_label"].to(device)
            strat_labels = torch.full((src.size(0),), -1, dtype=torch.long, device=device)

            out = model(src, tgt_in)
            losses = loss_fn(
                out["logits"], tgt_labels,
                out["emotion_logits"], emo_labels,
                out["strategy_logits"], strat_labels,
            )

            # Gradient accumulation: scale loss
            loss = losses["total"] / args.grad_accum
            loss.backward()
            accum_count += 1

            epoch_loss += losses["total"].item()
            epoch_gen  += losses["generation"].item()

            if accum_count == args.grad_accum or step_i == len(train_loader) - 1:
                # Gradient clipping — critical for training stability
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr = scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    ppl = compute_perplexity(epoch_gen / max(1, step_i + 1))
                    pbar.set_postfix({"ppl": f"{ppl:.1f}", "lr": f"{lr:.2e}"})
                    if args.use_wandb:
                        import wandb
                        wandb.log({"train/loss": losses["total"].item(), "train/ppl": ppl,
                                   "train/emotion_loss": losses["emotion"].item(), "lr": lr},
                                  step=global_step)

        # ── Validation ──────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        print(
            f"[Epoch {epoch:2d}] "
            f"train_ppl={compute_perplexity(epoch_gen/len(train_loader)):.1f}  "
            f"val_ppl={val_metrics['perplexity']:.1f}  "
            f"val_emo_loss={val_metrics['emotion']:.4f}"
        )

        if args.use_wandb:
            import wandb
            wandb.log({
                "val/ppl": val_metrics["perplexity"],
                "val/generation_loss": val_metrics["generation"],
                "val/emotion_loss": val_metrics["emotion"],
                "epoch": epoch,
            }, step=global_step)

        # ── Checkpoint ──────────────────────────────────────────────────
        if val_metrics["perplexity"] < best_val_ppl:
            best_val_ppl = val_metrics["perplexity"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                {"val_ppl": best_val_ppl},
                os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            print(f"  [patience {patience_counter}/{args.patience}]")
            if patience_counter >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, epoch, global_step,
        {"val_ppl": val_metrics["perplexity"]},
        os.path.join(args.checkpoint_dir, "final_model.pt"),
    )
    print(f"\n[Phase 1] Done. Best val perplexity: {best_val_ppl:.2f}")


if __name__ == "__main__":
    main()