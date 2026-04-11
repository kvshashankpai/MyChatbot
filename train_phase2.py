"""
train_phase2.py — Phase 2: Task-Specific Fine-Tuning on ESConv

Goal (Section 4.2):
  Specialise the Phase 1 model to therapeutic protocols, strategy-conditioned
  generation, and high-intensity emotional crisis contexts.

Key differences from Phase 1:
  • Peak LR: 2e-5 (10× lower — prevents catastrophic forgetting)
  • Warmup: 100 steps (shorter — model already knows language)
  • 30-50 epochs (small dataset needs more passes)
  • Strategy loss weight = 0.3 (NOW ACTIVE, class-weighted)
  • Teacher forcing: scheduled decay 100% → 50%
  • Primary early-stopping criterion: Distinct-2 score (not perplexity)
  • Strategy class weights from frequency distribution (Section 2.5)

Run command:
  python train_phase2.py \
    --phase1_checkpoint checkpoints/phase1/best_model.pt \
    --data_dir data/processed \
    --tokenizer_dir data/tokenizer \
    --checkpoint_dir checkpoints/phase2 \
    --epochs 40 \
    --use_wandb
"""

import argparse
import os
import sys
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from tokenizer import (
    HappyBotTokenizer,
    NUM_EMOTION_CLASSES_PHASE2,
    NUM_STRATEGY_CLASSES,
    ID_TO_STRATEGY,
)
from dataset import (
    HappyBotDataset,
    build_dataloader,
    compute_strategy_class_weights,
)
from model.transformer import HappyBot
from utils import (
    get_logger,
    get_warmup_cosine_scheduler,
    build_generation_loss,
    build_classification_loss,
    compute_total_loss,
    compute_perplexity,
    compute_distinct_n,
    compute_f1_score,
    get_teacher_forcing_ratio,
    build_optimizer,
    save_checkpoint,
    load_checkpoint,
    GradAccumulator,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Happy-Bot Phase 2 Fine-Tuning")
    p.add_argument("--phase1_checkpoint", required=True,
                   help="Path to best Phase 1 checkpoint")
    p.add_argument("--data_dir",          default="data/processed")
    p.add_argument("--tokenizer_dir",     default="data/tokenizer")
    p.add_argument("--checkpoint_dir",    default="checkpoints/phase2")
    p.add_argument("--log_dir",           default="logs")
    p.add_argument("--strategy_counts",   default="data/processed/strategy_counts.json",
                   help="JSON file with strategy frequency counts for class weighting")
    p.add_argument("--epochs",            type=int,   default=40)
    p.add_argument("--micro_batch",       type=int,   default=32)
    p.add_argument("--accum_steps",       type=int,   default=4)
    p.add_argument("--peak_lr",           type=float, default=2e-5)
    p.add_argument("--warmup_steps",      type=int,   default=100)
    p.add_argument("--weight_decay",      type=float, default=0.01)
    p.add_argument("--max_src_len",       type=int,   default=512)
    p.add_argument("--max_tgt_len",       type=int,   default=128)
    p.add_argument("--d_model",           type=int,   default=256)
    p.add_argument("--num_heads",         type=int,   default=2)
    p.add_argument("--num_layers",        type=int,   default=4)
    p.add_argument("--d_ff",              type=int,   default=1024)
    p.add_argument("--dropout",           type=float, default=0.1)
    p.add_argument("--label_smooth",      type=float, default=0.1)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--tf_start",          type=float, default=1.0,
                   help="Teacher forcing ratio at epoch 0")
    p.add_argument("--tf_end",            type=float, default=0.5,
                   help="Teacher forcing ratio at final epoch")
    p.add_argument("--log_every",         type=int,   default=50)
    p.add_argument("--eval_every",        type=int,   default=200)
    p.add_argument("--patience",          type=int,   default=10,
                   help="Early stopping patience (epochs without Distinct-2 improvement)")
    p.add_argument("--use_wandb",         action="store_true")
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Validation with Distinct-2 (primary metric, Section 4.4)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: HappyBot,
    val_loader: DataLoader,
    gen_loss_fn,
    emotion_loss_fn,
    strategy_loss_fn,
    device: torch.device,
    tokenizer: HappyBotTokenizer,
    max_batches: int = 100,
) -> dict:
    """
    Full validation loop.

    Computes:
      • Validation loss (total + per-component)
      • Perplexity
      • Emotion head weighted F1
      • Strategy head weighted F1
      • Distinct-2 on greedy-decoded samples
    """
    model.eval()

    total_metrics = {
        "val_loss": 0.0, "L_gen": 0.0, "L_emotion": 0.0, "L_strategy": 0.0
    }
    emotion_preds, emotion_tgts = [], []
    strategy_preds, strategy_tgts = [], []
    generated_texts = []
    n_batches = 0

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
            strategy_loss_fn=strategy_loss_fn,
            lambda_emotion=0.3,
            lambda_strategy=0.3,    # ACTIVE in Phase 2
        )

        for k in total_metrics:
            total_metrics[k] += loss_dict.get(k.replace("val_", "L_"), 0.0)
        total_metrics["val_loss"] += loss_dict["L_total"]

        emotion_preds.extend(out["emotion_logits"].argmax(-1).cpu().tolist())
        emotion_tgts.extend(batch["emotion_label"].cpu().tolist())
        strategy_preds.extend(out["strategy_logits"].argmax(-1).cpu().tolist())
        strategy_tgts.extend(batch["strategy_label"].cpu().tolist())

        # Greedy decode first sample in batch for Distinct-2
        if len(generated_texts) < 200:
            greedy_ids = out["gen_logits"][0].argmax(-1).cpu().tolist()
            text = tokenizer.decode(greedy_ids, skip_special_tokens=True)
            generated_texts.append(text)

        n_batches += 1

    avg_metrics = {k: v / n_batches for k, v in total_metrics.items()}
    avg_metrics["perplexity"] = compute_perplexity(avg_metrics["L_gen"])
    avg_metrics["emotion_f1"] = compute_f1_score(
        emotion_preds,
        [t for t in emotion_tgts if t >= 0],
        num_classes=NUM_EMOTION_CLASSES_PHASE2,
    )
    avg_metrics["strategy_f1"] = compute_f1_score(
        [p for p, t in zip(strategy_preds, strategy_tgts) if t >= 0],
        [t for t in strategy_tgts if t >= 0],
        num_classes=NUM_STRATEGY_CLASSES,
    )
    avg_metrics["distinct2"] = compute_distinct_n(generated_texts, n=2)
    avg_metrics["distinct1"] = compute_distinct_n(generated_texts, n=1)

    model.train()
    return avg_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Scheduled teacher forcing forward pass
# ─────────────────────────────────────────────────────────────────────────────

def forward_with_teacher_forcing(
    model: HappyBot,
    batch: dict,
    tf_ratio: float,
    device: torch.device,
):
    """
    Forward pass with scheduled teacher forcing.

    tf_ratio = 1.0: pure teacher forcing (feed ground truth at every step)
    tf_ratio = 0.5: 50% chance to use model's own prediction as next input

    WHY scheduled teacher forcing (Section 4.2):
      During training, the model always sees ground-truth tokens (no errors).
      At inference, it uses its own predictions. This mismatch (exposure bias)
      causes compounding errors at generation time. Scheduled sampling
      gradually bridges this gap.

    TRICKY: We still process the full sequence in parallel (batch), but
    with a probability (1 - tf_ratio) we replace teacher-forced positions
    with the model's top-1 prediction from the previous forward pass.
    This is an approximation of true scheduled sampling.
    """
    if tf_ratio >= 1.0:
        # Pure teacher forcing — standard forward pass
        return model(
            encoder_ids=batch["encoder_ids"],
            decoder_input=batch["decoder_input"],
            encoder_mask=batch["encoder_mask"],
            decoder_mask=batch["decoder_mask"],
        )

    # ── Approximate scheduled sampling ────────────────────────────────────
    # First, run a full teacher-forced forward to get per-position predictions
    with torch.no_grad():
        out_tf = model(
            encoder_ids=batch["encoder_ids"],
            decoder_input=batch["decoder_input"],
            encoder_mask=batch["encoder_mask"],
            decoder_mask=batch["decoder_mask"],
        )
        predicted_ids = out_tf["gen_logits"].argmax(dim=-1)  # (B, T)

    # Build mixed decoder input: some positions use ground truth, some use predictions
    # Sampling mask: True = use model prediction, False = use ground truth
    sampling_mask = (torch.rand(batch["decoder_input"].shape, device=device) > tf_ratio)

    # Shift predictions right by one position (prediction at t predicts t+1 input)
    # Keep [BOS] token at position 0 always from ground truth
    mixed_input = batch["decoder_input"].clone()
    if batch["decoder_input"].size(1) > 1:
        # predicted_ids[0:T-1] provides t+1 inputs; position 0 always = BOS
        shifted_preds = predicted_ids[:, :-1]  # (B, T-1)
        mask_subset   = sampling_mask[:, 1:]   # (B, T-1) skip position 0
        mixed_input[:, 1:][mask_subset] = shifted_preds[mask_subset]

    # Final forward with mixed input
    return model(
        encoder_ids=batch["encoder_ids"],
        decoder_input=mixed_input,
        encoder_mask=batch["encoder_mask"],
        decoder_mask=batch["decoder_mask"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    import json
    from collections import Counter

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger("phase2", log_file=f"{args.log_dir}/phase2.log")
    logger.info(f"Device: {device}")
    logger.info(f"Args: {vars(args)}")

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project="happy-bot", name="phase2", config=vars(args))
        except ImportError:
            logger.warning("wandb not installed.")
            args.use_wandb = False

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer  = HappyBotTokenizer(args.tokenizer_dir)
    vocab_size = tokenizer.vocab_size
    pad_id     = tokenizer.pad_id

    # ── Strategy class weights ─────────────────────────────────────────────
    # Load from pre-computed counts (produced during dataset extraction)
    if os.path.exists(args.strategy_counts):
        with open(args.strategy_counts) as f:
            raw_counts = json.load(f)
        strategy_counter = Counter(raw_counts)
    else:
        # Fallback: use approximate frequencies from Section 2.5 of spec
        logger.warning("strategy_counts.json not found, using spec frequencies.")
        strategy_counter = Counter({
            "question": 207, "restatement": 152, "affirmation": 148,
            "reflection": 115, "suggestion": 98, "information": 93,
            "selfdisclosure": 87, "other": 100,
        })

    strategy_weights = compute_strategy_class_weights(
        strategy_counter, num_classes=NUM_STRATEGY_CLASSES, device=device
    )

    # ── Datasets ───────────────────────────────────────────────────────────
    train_path = os.path.join(args.data_dir, "esconv_train.jsonl")
    val_path   = os.path.join(args.data_dir, "esconv_val.jsonl")

    train_ds = HappyBotDataset(train_path, tokenizer,
                                args.max_src_len, args.max_tgt_len, phase=2)
    val_ds   = HappyBotDataset(val_path,   tokenizer,
                                args.max_src_len, args.max_tgt_len, phase=2)

    train_loader = build_dataloader(train_ds, args.micro_batch, pad_id, shuffle=True)
    val_loader   = build_dataloader(val_ds,   args.micro_batch, pad_id, shuffle=False)

    # ── Model (initialise from Phase 1 checkpoint) ─────────────────────────
    model = HappyBot(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        # Phase 2 uses 8-class ESConv emotions for the emotion head
        num_emotion_classes=NUM_EMOTION_CLASSES_PHASE2,
        num_strategy_classes=NUM_STRATEGY_CLASSES,
        pad_token_id=pad_id,
    ).to(device)

    # Load Phase 1 weights — only load compatible keys (emotion head may differ)
    logger.info(f"Loading Phase 1 checkpoint: {args.phase1_checkpoint}")
    ckpt = torch.load(args.phase1_checkpoint, map_location=device)
    state = ckpt["model_state"]
    # Partial load: skip emotion head if class count changed
    model_state = model.state_dict()
    compat_state = {
        k: v for k, v in state.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    skipped = set(state.keys()) - set(compat_state.keys())
    if skipped:
        logger.info(f"Skipped {len(skipped)} incompatible keys: {list(skipped)[:5]}")
    model_state.update(compat_state)
    model.load_state_dict(model_state)
    logger.info(f"Phase 1 weights loaded ({len(compat_state)} / {len(state)} keys).")

    # ── Loss functions ─────────────────────────────────────────────────────
    gen_loss_fn      = build_generation_loss(vocab_size, pad_id, args.label_smooth)
    emotion_loss_fn  = build_classification_loss(class_weights=None)
    strategy_loss_fn = build_classification_loss(class_weights=strategy_weights)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    # IMPORTANT: Reset optimizer state — don't carry Phase 1 momentum into Phase 2
    optimizer   = build_optimizer(model, args.peak_lr, args.weight_decay)
    total_steps = (len(train_loader) // args.accum_steps) * args.epochs
    scheduler   = get_warmup_cosine_scheduler(optimizer, args.warmup_steps, total_steps)

    # ── Training loop ──────────────────────────────────────────────────────
    accum = GradAccumulator(steps=args.accum_steps)
    best_distinct2 = 0.0
    patience_count = 0
    global_step = 0

    logger.info(f"Starting Phase 2 fine-tuning: {args.epochs} epochs, "
                f"peak_lr={args.peak_lr}, warmup={args.warmup_steps}")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # Compute teacher forcing ratio for this epoch
        tf_ratio = get_teacher_forcing_ratio(
            epoch, start_epoch=0, total_epochs=args.epochs,
            start_ratio=args.tf_start, end_ratio=args.tf_end
        )
        logger.info(f"Epoch {epoch}: teacher_forcing_ratio={tf_ratio:.2f}")

        epoch_start = time.time()
        running_loss = {k: 0.0 for k in ["L_total", "L_gen", "L_emotion", "L_strategy"]}

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # ── Forward pass with scheduled teacher forcing ────────────────
            out = forward_with_teacher_forcing(model, batch, tf_ratio, device)

            # ── Unified multi-task loss ────────────────────────────────────
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
                lambda_strategy=0.3,   # ACTIVE
            )

            (L_total / args.accum_steps).backward()

            for k in running_loss:
                running_loss[k] += loss_dict.get(k, 0.0)

            if accum.should_step():
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.log_every == 0:
                    avg = {k: v / args.log_every for k, v in running_loss.items()}
                    lr  = scheduler.get_last_lr()[0]
                    logger.info(
                        f"E{epoch:03d} step={global_step:06d}  "
                        f"L={avg['L_total']:.4f}  "
                        f"L_gen={avg['L_gen']:.4f}  "
                        f"L_emo={avg['L_emotion']:.4f}  "
                        f"L_strat={avg['L_strategy']:.4f}  "
                        f"lr={lr:.2e}  tf={tf_ratio:.2f}"
                    )
                    if args.use_wandb:
                        import wandb
                        wandb.log({**{"train/" + k: v for k, v in avg.items()},
                                   "lr": lr, "tf_ratio": tf_ratio},
                                  step=global_step)
                    running_loss = {k: 0.0 for k in running_loss}

                if global_step % args.eval_every == 0:
                    val_metrics = evaluate(
                        model, val_loader,
                        gen_loss_fn, emotion_loss_fn, strategy_loss_fn,
                        device, tokenizer
                    )
                    logger.info(
                        f"  [VAL] ppl={val_metrics['perplexity']:.2f}  "
                        f"D1={val_metrics['distinct1']:.4f}  "
                        f"D2={val_metrics['distinct2']:.4f}  "
                        f"emo_f1={val_metrics['emotion_f1']:.4f}  "
                        f"strat_f1={val_metrics['strategy_f1']:.4f}"
                    )
                    if args.use_wandb:
                        import wandb
                        wandb.log({"val/" + k: v for k, v in val_metrics.items()},
                                  step=global_step)

                    # Save best model by Distinct-2 (primary metric, Section 4.4)
                    if val_metrics["distinct2"] > best_distinct2:
                        best_distinct2 = val_metrics["distinct2"]
                        patience_count = 0
                        save_checkpoint(
                            model, optimizer, scheduler,
                            epoch, global_step, best_distinct2,
                            metric_name="distinct2",
                            checkpoint_dir=args.checkpoint_dir,
                            filename="best_model.pt",
                        )
                    else:
                        patience_count += 1

        # Epoch-level early stopping check
        if patience_count >= args.patience:
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(no Distinct-2 improvement for {patience_count} eval steps)")
            break

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch} done in {epoch_time:.1f}s")

    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, global_step,
                    best_distinct2, "distinct2", args.checkpoint_dir,
                    filename="final_model.pt")

    logger.info(f"Phase 2 complete. Best Distinct-2: {best_distinct2:.4f}")
    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    train(parse_args())
