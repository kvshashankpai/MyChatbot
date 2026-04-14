"""
train_phase1.py — Phase 1: pre-train on EmpatheticDialogues.

Key Phase 1 settings:
  - lambda_strategy = 0.0  (ED has no strategy labels — MUST be zero)
  - lambda_emotion  = 0.3  (emotion labels are present)
  - Warmup 500 steps then cosine annealing
  - Gradient clip 1.0
  - Early stop on val perplexity, patience=8

FIXES vs original:
  1. Default --epochs raised from 10 → 30. Phase 1 was stopped too early:
     val_ppl was still improving at epoch 10 (123.06). Target is val_ppl < 60
     before Phase 2 fine-tuning begins.
  2. Default --patience raised from 5 → 8 to give the scheduler room to
     escape local plateaux.
  3. Best PPL printed every epoch so you can see exact improvement rate.
  4. Gradient norm logged every log_every steps so vanishing/exploding
     gradients are visible early.
"""

import argparse
import math
import os
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
    p.add_argument("--data_dir",        default="data/processed")
    p.add_argument("--tokenizer_dir",   default="data/tokenizer")
    p.add_argument("--checkpoint_dir",  default="checkpoints/phase1")
    p.add_argument("--epochs",          type=int,   default=30)    # FIX: was 20
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--grad_accum",      type=int,   default=4)
    p.add_argument("--peak_lr",         type=float, default=1e-4)
    p.add_argument("--warmup_steps",    type=int,   default=500)
    p.add_argument("--max_grad_norm",   type=float, default=1.0)
    p.add_argument("--weight_decay",    type=float, default=0.01)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--log_every",       type=int,   default=100)
    p.add_argument("--patience",        type=int,   default=8)     # FIX: was 5
    p.add_argument("--use_wandb",       action="store_true")
    return p.parse_args()


def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot_gen = tot_emo = n = 0
    with torch.no_grad():
        for batch in loader:
            src      = batch["encoder_ids"].to(device)
            tgt_in   = batch["decoder_input_ids"].to(device)
            tgt_lab  = batch["decoder_target_ids"].to(device)
            emo_lab  = batch["emotion_label"].to(device)
            # Phase 1: no strategy labels → pass -1 so ignore_index kicks in
            strat_lab = torch.full((src.size(0),), -1, dtype=torch.long, device=device)

            out    = model(src, tgt_in)
            losses = loss_fn(
                out["logits"], tgt_lab,
                out["emotion_logits"], emo_lab,
                out["strategy_logits"], strat_lab,
            )
            tot_gen += losses["generation"].item()
            tot_emo += losses["emotion"].item()
            n += 1

    return {
        "generation": tot_gen / max(1, n),
        "emotion":    tot_emo / max(1, n),
        "perplexity": compute_perplexity(tot_gen / max(1, n)),
    }


def main():
    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Phase 1] device={device}")

    from tokenizers import Tokenizer
    tok      = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    vocab_sz = tok.get_vocab_size()
    pad_id   = tok.token_to_id("[PAD]")
    print(f"[Phase 1] vocab={vocab_sz}  pad_id={pad_id}")

    train_ds = HappyBotDataset(os.path.join(args.data_dir, "empathetic_train.jsonl"), tok, phase=1)
    val_ds   = HappyBotDataset(os.path.join(args.data_dir, "empathetic_val.jsonl"),   tok, phase=1)
    cfn      = lambda b: collate_fn(b, pad_id=pad_id)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=cfn, num_workers=0)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=cfn, num_workers=0)
    print(f"[Phase 1] train={len(train_ds)}  val={len(val_ds)}")

    model = HappyBot(
        vocab_size=vocab_sz, d_model=256, num_encoder_layers=4, num_decoder_layers=4,
        num_heads=2, d_ff=1024, dropout=0.1,
        num_emotion_classes=32, num_strategy_classes=8, pad_token_id=pad_id,
    ).to(device)
    print(f"[Phase 1] params={model.count_parameters():,}")

    # CRITICAL: lambda_strategy = 0.0 in Phase 1 — ED has no strategy labels
    loss_fn = MultiTaskLoss(
        vocab_size=vocab_sz,
        label_smoothing=args.label_smoothing,
        lambda_emotion=0.3,
        lambda_strategy=0.0,
    )

    optimizer = build_optimizer(model, args.peak_lr, args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps     = args.epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps, args.peak_lr)

    if args.use_wandb:
        import wandb
        wandb.init(project="happybot", name="phase1", config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_ppl    = float("inf")
    patience_c  = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        accum  = 0
        ep_gen = ep_emo = 0.0

        pbar = tqdm(train_loader, desc=f"Ep{epoch}/{args.epochs}", leave=False)
        for i, batch in enumerate(pbar):
            src      = batch["encoder_ids"].to(device)
            tgt_in   = batch["decoder_input_ids"].to(device)
            tgt_lab  = batch["decoder_target_ids"].to(device)
            emo_lab  = batch["emotion_label"].to(device)
            strat_lab = torch.full((src.size(0),), -1, dtype=torch.long, device=device)

            out    = model(src, tgt_in)
            losses = loss_fn(
                out["logits"], tgt_lab,
                out["emotion_logits"], emo_lab,
                out["strategy_logits"], strat_lab,
            )

            (losses["total"] / args.grad_accum).backward()
            accum  += 1
            ep_gen += losses["generation"].item()
            ep_emo += losses["emotion"].item()

            if accum == args.grad_accum or i == len(train_loader) - 1:
                # FIX: capture grad norm before clipping for diagnostics
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr = scheduler.step()
                optimizer.zero_grad()
                accum = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    ppl = compute_perplexity(ep_gen / max(1, i + 1))
                    pbar.set_postfix(ppl=f"{ppl:.1f}", gnorm=f"{grad_norm:.2f}", lr=f"{lr:.2e}")
                    if args.use_wandb:
                        import wandb
                        wandb.log({
                            "train/ppl": ppl, "train/emo_loss": losses["emotion"].item(),
                            "train/grad_norm": grad_norm, "lr": lr,
                        }, step=global_step)

        val = evaluate(model, val_loader, loss_fn, device)
        train_ppl = compute_perplexity(ep_gen / len(train_loader))
        print(
            f"[Ep {epoch:2d}] train_ppl={train_ppl:.1f}"
            f"  val_ppl={val['perplexity']:.1f}"
            f"  val_emo={val['emotion']:.4f}"
            f"  best_ppl={best_ppl:.2f}"   # FIX: show best every epoch
        )

        if args.use_wandb:
            import wandb
            wandb.log({"val/ppl": val["perplexity"], "val/emo_loss": val["emotion"],
                       "epoch": epoch}, step=global_step)

        if val["perplexity"] < best_ppl:
            best_ppl   = val["perplexity"]
            patience_c = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                {"val_ppl": best_ppl},
                os.path.join(args.checkpoint_dir, "best_model.pt"),
            )
            print(f"  ✓ New best val_ppl={best_ppl:.2f} — checkpoint saved")
        else:
            patience_c += 1
            print(f"  patience {patience_c}/{args.patience}  best_ppl={best_ppl:.2f}")
            if patience_c >= args.patience:
                print("  Early stopping.")
                break

    save_checkpoint(
        model, optimizer, scheduler, epoch, global_step,
        {"val_ppl": val["perplexity"]},
        os.path.join(args.checkpoint_dir, "final_model.pt"),
    )
    print(f"\n[Phase 1] Done. Best val ppl: {best_ppl:.2f}")
    if best_ppl > 60:
        print(
            f"[Phase 1] WARNING: best_ppl={best_ppl:.1f} > 60. "
            "Consider running more epochs before Phase 2 fine-tuning."
        )


if __name__ == "__main__":
    main()