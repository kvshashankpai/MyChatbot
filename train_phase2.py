"""
train_phase2.py — Phase 2: fine-tune on ESConv.

Key differences from Phase 1:
  - Loads Phase 1 best checkpoint
  - Peak LR = 2e-5  (10× lower — prevents catastrophic forgetting)
  - lambda_strategy = 0.3 (now active with class-weighted loss)
  - Early stop on Distinct-2 (diversity), not perplexity
"""

import argparse
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.transformer import HappyBot
from dataset import HappyBotDataset, collate_fn, CANONICAL_STRATEGIES
from utils import (
    MultiTaskLoss,
    WarmupCosineScheduler,
    build_optimizer,
    compute_perplexity,
    compute_distinct_n,
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
    p.add_argument("--peak_lr",           type=float, default=2e-5)
    p.add_argument("--warmup_steps",      type=int,   default=100)
    p.add_argument("--max_grad_norm",     type=float, default=1.0)
    p.add_argument("--weight_decay",      type=float, default=0.01)
    p.add_argument("--tf_start",          type=float, default=1.0)
    p.add_argument("--tf_end",            type=float, default=0.5)
    p.add_argument("--log_every",         type=int,   default=50)
    p.add_argument("--patience",          type=int,   default=8)
    p.add_argument("--use_wandb",         action="store_true")
    return p.parse_args()


def evaluate(model, loader, loss_fn, tokenizer, device, max_gen=200):
    model.eval()
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")
    pad = tokenizer.token_to_id("[PAD]")

    tot_gen = tot_emo = tot_str = n = 0
    em_p, em_t, st_p, st_t = [], [], [], []
    gen_seqs = []

    with torch.no_grad():
        for batch in loader:
            src     = batch["encoder_ids"].to(device)
            tgt_in  = batch["decoder_input_ids"].to(device)
            tgt_lab = batch["decoder_target_ids"].to(device)
            emo_lab = batch["emotion_label"].to(device)
            str_lab = batch["strategy_label"].to(device)

            out = model(src, tgt_in)
            losses = loss_fn(out["logits"], tgt_lab,
                             out["emotion_logits"], emo_lab,
                             out["strategy_logits"], str_lab)
            tot_gen += losses["generation"].item()
            tot_emo += losses["emotion"].item()
            tot_str += losses["strategy"].item()
            n += 1

            em_p.extend(out["emotion_logits"].argmax(-1).cpu().tolist())
            em_t.extend(emo_lab.cpu().tolist())
            st_p.extend(out["strategy_logits"].argmax(-1).cpu().tolist())
            st_t.extend(str_lab.cpu().tolist())

            # Greedy generation for diversity metrics
            if len(gen_seqs) < max_gen:
                enc   = model.encode(src)
                mem   = enc["memory"]
                smask = enc["src_mask"]
                cur   = torch.full((src.size(0), 1), bos, dtype=torch.long, device=device)
                for _ in range(64):
                    nxt = model.decode_step(cur, mem, smask).argmax(-1, keepdim=True)
                    cur = torch.cat([cur, nxt], 1)
                    if (nxt == eos).all(): break
                for seq in cur.cpu().tolist():
                    gen_seqs.append([t for t in seq[1:] if t not in (bos, eos, pad)])

    def wf1(preds, targets):
        valid = [(p, t) for p, t in zip(preds, targets) if t != -1]
        if not valid: return 0.0
        ps, ts = zip(*valid)
        classes = set(ts)
        tot = len(ts)
        f1 = 0.0
        for c in classes:
            tp = sum(1 for p, t in zip(ps, ts) if p == c == t)
            fp = sum(1 for p, t in zip(ps, ts) if p == c and t != c)
            fn = sum(1 for p, t in zip(ps, ts) if p != c and t == c)
            pr = tp / (tp + fp + 1e-8); rc = tp / (tp + fn + 1e-8)
            f1 += 2*pr*rc/(pr+rc+1e-8) * sum(1 for t in ts if t==c) / tot
        return f1

    avg = lambda x: x / max(1, n)
    return {
        "perplexity":  compute_perplexity(avg(tot_gen)),
        "generation":  avg(tot_gen),
        "emotion":     avg(tot_emo),
        "strategy":    avg(tot_str),
        "distinct1":   compute_distinct_n(gen_seqs, 1),
        "distinct2":   compute_distinct_n(gen_seqs, 2),
        "emotion_f1":  wf1(em_p, em_t),
        "strategy_f1": wf1(st_p, st_t),
    }


def main():
    args   = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Phase 2] device={device}")

    from tokenizers import Tokenizer
    tok    = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    vocab_sz = tok.get_vocab_size()
    pad_id   = tok.token_to_id("[PAD]")

    train_ds = HappyBotDataset(os.path.join(args.data_dir, "esconv_train.jsonl"), tok, phase=2)
    val_ds   = HappyBotDataset(os.path.join(args.data_dir, "esconv_val.jsonl"),   tok, phase=2)
    cfn      = lambda b: collate_fn(b, pad_id=pad_id)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  collate_fn=cfn, num_workers=0)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, collate_fn=cfn, num_workers=0)
    print(f"[Phase 2] train={len(train_ds)}  val={len(val_ds)}")

    sw = compute_strategy_weights(
        os.path.join(args.data_dir, "strategy_counts.json"), num_classes=8, device=device)
    print(f"[Phase 2] strategy weights: {sw.tolist()}")

    model = HappyBot(
        vocab_size=vocab_sz, d_model=256, num_encoder_layers=4, num_decoder_layers=4,
        num_heads=2, d_ff=1024, dropout=0.1,
        num_emotion_classes=32, num_strategy_classes=8, pad_token_id=pad_id,
    ).to(device)

    print(f"[Phase 2] Loading Phase 1 ckpt: {args.phase1_checkpoint}")
    ckpt = load_checkpoint(args.phase1_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    loss_fn = MultiTaskLoss(
        vocab_size=vocab_sz, label_smoothing=0.1,
        lambda_emotion=0.3,
        lambda_strategy=0.3,      # ← NOW ACTIVE
        strategy_weights=sw,
    )

    optimizer = build_optimizer(model, args.peak_lr, args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps     = args.epochs * steps_per_epoch
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_steps, total_steps, args.peak_lr)

    if args.use_wandb:
        import wandb
        wandb.init(project="happybot", name="phase2", config=vars(args))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_d2   = -1.0
    patience_c = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        accum = 0
        ep_loss = 0.0

        # Teacher forcing ratio: linear decay
        tf_ratio = args.tf_start + (args.tf_end - args.tf_start) * (epoch - 1) / max(1, args.epochs - 1)

        pbar = tqdm(train_loader, desc=f"P2 Ep{epoch}/{args.epochs}", leave=False)
        for i, batch in enumerate(pbar):
            src     = batch["encoder_ids"].to(device)
            tgt_in  = batch["decoder_input_ids"].to(device)
            tgt_lab = batch["decoder_target_ids"].to(device)
            emo_lab = batch["emotion_label"].to(device)
            str_lab = batch["strategy_label"].to(device)

            out    = model(src, tgt_in)
            losses = loss_fn(out["logits"], tgt_lab,
                             out["emotion_logits"], emo_lab,
                             out["strategy_logits"], str_lab)

            (losses["total"] / args.grad_accum).backward()
            accum   += 1
            ep_loss += losses["total"].item()

            if accum == args.grad_accum or i == len(train_loader) - 1:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr = scheduler.step()
                optimizer.zero_grad()
                accum = 0
                global_step += 1

                if global_step % args.log_every == 0:
                    pbar.set_postfix(loss=f"{losses['total'].item():.3f}",
                                     tf=f"{tf_ratio:.2f}", lr=f"{lr:.2e}")

        val = evaluate(model, val_loader, loss_fn, tok, device)
        print(f"[Ep {epoch:2d}] ppl={val['perplexity']:.1f}"
              f"  D1={val['distinct1']:.3f}  D2={val['distinct2']:.3f}"
              f"  emo_f1={val['emotion_f1']:.3f}  str_f1={val['strategy_f1']:.3f}"
              f"  tf={tf_ratio:.2f}")

        if args.use_wandb:
            import wandb
            wandb.log({**{"epoch": epoch, "tf_ratio": tf_ratio},
                       **{f"val/{k}": v for k, v in val.items()}}, step=global_step)

        if val["distinct2"] > best_d2:
            best_d2    = val["distinct2"]
            patience_c = 0
            save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                            val, os.path.join(args.checkpoint_dir, "best_model.pt"))
        else:
            patience_c += 1
            print(f"  patience {patience_c}/{args.patience}  best_D2={best_d2:.3f}")
            if patience_c >= args.patience:
                print("  Early stopping.")
                break

    save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                    val, os.path.join(args.checkpoint_dir, "final_model.pt"))
    print(f"\n[Phase 2] Done. Best D2={best_d2:.3f}")


if __name__ == "__main__":
    main()