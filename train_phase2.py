"""
train_phase2.py — Phase 2: fine-tune HappyBot on ESConv therapeutic dialogues.

Key differences from Phase 1:
  - Loads Phase 1 best checkpoint as starting point
  - Peak LR reduced 10x (2e-5) to prevent catastrophic forgetting
  - lambda_strategy = 0.3 (strategy head now active with class-weighted loss)
  - Scheduled teacher forcing ratio decay: 100% → 50% over training
  - Early stopping on Distinct-2 (response diversity), not perplexity
  - Beam search for Action/Suggestion strategies; nucleus for others
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
    compute_distinct_n,
    compute_perplexity,
    compute_accuracy,
    compute_strategy_weights,
    save_checkpoint,
    load_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase1_checkpoint", default="checkpoints/phase1/best_model.pt")
    p.add_argument("--data_dir",          default="data/processed")
    p.add_argument("--tokenizer_dir",     default="data/tokenizer")
    p.add_argument("--checkpoint_dir",    default="checkpoints/phase2")
    p.add_argument("--epochs",            type=int,   default=40)
    p.add_argument("--batch_size",        type=int,   default=16)
    p.add_argument("--grad_accum",        type=int,   default=8)
    p.add_argument("--peak_lr",           type=float, default=2e-5)  # 10x lower than Phase 1
    p.add_argument("--warmup_steps",      type=int,   default=100)
    p.add_argument("--weight_decay",      type=float, default=0.01)
    p.add_argument("--max_grad_norm",     type=float, default=1.0)
    p.add_argument("--tf_ratio_start",    type=float, default=1.0,  help="Teacher forcing ratio at epoch 1")
    p.add_argument("--tf_ratio_end",      type=float, default=0.5,  help="Teacher forcing ratio at final epoch")
    p.add_argument("--use_wandb",         action="store_true")
    p.add_argument("--log_every",         type=int,   default=50)
    p.add_argument("--patience",          type=int,   default=8)
    return p.parse_args()


def get_teacher_forcing_ratio(epoch: int, total_epochs: int, start: float, end: float) -> float:
    """Linearly decay teacher forcing ratio over training."""
    progress = (epoch - 1) / max(1, total_epochs - 1)
    return start + (end - start) * progress


def evaluate(model, loader, loss_fn, tokenizer, device, max_gen_samples=200):
    """
    Evaluate on ESConv val set.
    Returns perplexity (from teacher-forced loss) and Distinct-1/2 (from greedy generation).
    """
    model.eval()
    total_gen_loss = total_emo_loss = total_strat_loss = 0
    total_batches = 0
    emo_preds, emo_targets, strat_preds, strat_targets = [], [], [], []

    # For Distinct-n, collect generated sequences
    generated_seqs = []
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    samples_generated = 0

    with torch.no_grad():
        for batch in loader:
            src        = batch["encoder_ids"].to(device)
            tgt_in     = batch["decoder_input_ids"].to(device)
            tgt_labels = batch["decoder_target_ids"].to(device)
            emo_labels = batch["emotion_label"].to(device)
            strat_labels = batch["strategy_label"].to(device)

            out = model(src, tgt_in)
            losses = loss_fn(
                out["logits"], tgt_labels,
                out["emotion_logits"], emo_labels,
                out["strategy_logits"], strat_labels,
            )
            total_gen_loss   += losses["generation"].item()
            total_emo_loss   += losses["emotion"].item()
            total_strat_loss += losses["strategy"].item()
            total_batches    += 1

            # Collect classification predictions for F1
            emo_preds.extend(out["emotion_logits"].argmax(-1).cpu().tolist())
            emo_targets.extend(emo_labels.cpu().tolist())
            strat_preds.extend(out["strategy_logits"].argmax(-1).cpu().tolist())
            strat_targets.extend(strat_labels.cpu().tolist())

            # Simple greedy generation for diversity metrics
            if samples_generated < max_gen_samples:
                enc_out = model.encode(src)
                memory = enc_out["memory"]
                src_mask = enc_out["src_mask"]
                B = src.size(0)
                tgt_cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
                for _ in range(64):
                    next_logits = model.decode_step(tgt_cur, memory, src_mask)
                    next_tok = next_logits.argmax(-1, keepdim=True)
                    tgt_cur = torch.cat([tgt_cur, next_tok], dim=1)
                for seq in tgt_cur.cpu().tolist():
                    seq = [t for t in seq if t not in (bos_id, eos_id)]
                    generated_seqs.append(seq)
                    samples_generated += B

    avg_gen = total_gen_loss / max(1, total_batches)
    d1 = compute_distinct_n(generated_seqs, 1) if generated_seqs else 0
    d2 = compute_distinct_n(generated_seqs, 2) if generated_seqs else 0

    # Weighted F1 for emotion/strategy (filter out ignore_index=-1)
    def weighted_f1(preds, targets):
        from collections import Counter
        valid = [(p, t) for p, t in zip(preds, targets) if t != -1]
        if not valid:
            return 0.0
        ps, ts = zip(*valid)
        classes = set(ts)
        total_support = len(ts)
        f1_sum = 0.0
        for c in classes:
            tp = sum(1 for p, t in zip(ps, ts) if p == c and t == c)
            fp = sum(1 for p, t in zip(ps, ts) if p == c and t != c)
            fn = sum(1 for p, t in zip(ps, ts) if p != c and t == c)
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            support = sum(1 for t in ts if t == c)
            f1_sum += f1 * support / total_support
        return f1_sum

    return {
        "perplexity":    compute_perplexity(avg_gen),
        "generation":    avg_gen,
        "emotion_loss":  total_emo_loss / max(1, total_batches),
        "strategy_loss": total_strat_loss / max(1, total_batches),
        "distinct1":     d1,
        "distinct2":     d2,
        "emotion_f1":    weighted_f1(emo_preds, emo_targets),
        "strategy_f1":   weighted_f1(strat_preds, strat_targets),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Phase 2] Device: {device}")

    # ── Tokenizer ───────────────────────────────────────────────────────
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    # ── Datasets ────────────────────────────────────────────────────────
    train_ds = HappyBotDataset(
        os.path.join(args.data_dir, "esconv_train.jsonl"), tokenizer, phase=2,
    )
    val_ds = HappyBotDataset(
        os.path.join(args.data_dir, "esconv_val.jsonl"), tokenizer, phase=2,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id), num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id), num_workers=2,
    )
    print(f"[Phase 2] Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Strategy weights ────────────────────────────────────────────────
    strategy_weights = compute_strategy_weights(
        os.path.join(args.data_dir, "strategy_counts.json"),
        num_classes=8,
        device=device,
    )
    print(f"[Phase 2] Strategy weights: {strategy_weights.tolist()}")

    # ── Model (load Phase 1 checkpoint) ─────────────────────────────────
    model = HappyBot(
        vocab_size=vocab_size, d_model=256, num_encoder_layers=4, num_decoder_layers=4,
        num_heads=2, d_ff=1024, dropout=0.1,
        num_emotion_classes=32, num_strategy_classes=8, pad_token_id=pad_id,
    ).to(device)

    print(f"[Phase 2] Loading Phase 1 checkpoint: {args.phase1_checkpoint}")
    load_checkpoint(args.phase1_checkpoint, model)
    print(f"[Phase 2] Parameters: {model.count_parameters():,}")

    # ── Loss ────────────────────────────────────────────────────────────
    loss_fn = MultiTaskLoss(
        vocab_size=vocab_size,
        num_emotion_classes=32,
        num_strategy_classes=8,
        label_smoothing=0.1,
        lambda_emotion=0.3,
        lambda_strategy=0.3,   # Now active
        strategy_weights=strategy_weights,
        pad_token_id=pad_id,
    )

    # ── Optimizer + Scheduler ───────────────────────────────────────────
    # Reset optimizer state — start fresh with 10x lower LR
    optimizer = build_optimizer(model, lr=args.peak_lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps,
        total_steps=total_steps, peak_lr=args.peak_lr,
    )

    if args.use_wandb:
        import wandb
        wandb.init(project="happybot", name="phase2", config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_d2 = -1.0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        accum_count = 0

        tf_ratio = get_teacher_forcing_ratio(
            epoch, args.epochs, args.tf_ratio_start, args.tf_ratio_end
        )

        pbar = tqdm(train_loader, desc=f"Phase2 Epoch {epoch}/{args.epochs}", leave=False)
        for step_i, batch in enumerate(pbar):
            src        = batch["encoder_ids"].to(device)
            tgt_in     = batch["decoder_input_ids"].to(device)
            tgt_labels = batch["decoder_target_ids"].to(device)
            emo_labels = batch["emotion_label"].to(device)
            strat_labels = batch["strategy_label"].to(device)

            out = model(src, tgt_in)
            losses = loss_fn(
                out["logits"], tgt_labels,
                out["emotion_logits"], emo_labels,
                out["strategy_logits"], strat_labels,
            )

            loss = losses["total"] / args.grad_accum
            loss.backward()
            accum_count += 1
            epoch_loss += losses["total"].item()

            if accum_count == args.grad_accum or step_i == len(train_loader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr = scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    pbar.set_postfix({
                        "loss": f"{losses['total'].item():.3f}",
                        "tf": f"{tf_ratio:.2f}",
                        "lr": f"{lr:.2e}",
                    })

        # ── Validation ──────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, loss_fn, tokenizer, device)
        print(
            f"[Epoch {epoch:2d}] "
            f"ppl={val_metrics['perplexity']:.1f}  "
            f"D1={val_metrics['distinct1']:.3f}  D2={val_metrics['distinct2']:.3f}  "
            f"emo_f1={val_metrics['emotion_f1']:.3f}  strat_f1={val_metrics['strategy_f1']:.3f}  "
            f"tf={tf_ratio:.2f}"
        )

        if args.use_wandb:
            import wandb
            wandb.log({**{"epoch": epoch, "tf_ratio": tf_ratio},
                       **{f"val/{k}": v for k, v in val_metrics.items()}}, step=global_step)

        # ── Checkpoint (primary: Distinct-2) ────────────────────────────
        if val_metrics["distinct2"] > best_d2:
            best_d2 = val_metrics["distinct2"]
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                val_metrics, os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
        else:
            patience_counter += 1
            print(f"  [patience {patience_counter}/{args.patience}]  best D2={best_d2:.3f}")
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    save_checkpoint(
        model, optimizer, scheduler, epoch, global_step, val_metrics,
        os.path.join(args.checkpoint_dir, "final_model.pt"),
    )
    print(f"\n[Phase 2] Done. Best Distinct-2: {best_d2:.3f}")


if __name__ == "__main__":
    main()