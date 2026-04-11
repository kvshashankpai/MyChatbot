"""
scripts/evaluate.py — Full evaluation suite for HappyBot.

Computes:
  - Perplexity (generation loss)
  - Distinct-1 and Distinct-2 (response diversity)
  - Weighted F1 for Emotion Head and Strategy Head
  - Confusion matrices for both NLU heads
  - Attention weight heatmaps (encoder CLS + decoder cross-attention)
  - Sample response generation on held-out test examples

Usage:
  python scripts/evaluate.py \
    --checkpoint checkpoints/phase2/best_model.pt \
    --tokenizer_dir data/tokenizer \
    --test_data data/processed/esconv_test.jsonl \
    --output_dir outputs/evaluation
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from model.transformer import HappyBot
from dataset import HappyBotDataset, collate_fn, CANONICAL_STRATEGIES
from utils import compute_distinct_n, compute_perplexity, load_checkpoint
from inference import top_p_sampling, EMOTION_LABELS_ED


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--tokenizer_dir",  default="data/tokenizer")
    p.add_argument("--test_data",      default="data/processed/esconv_test.jsonl")
    p.add_argument("--output_dir",     default="outputs/evaluation")
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--max_gen_len",    type=int, default=128)
    p.add_argument("--num_samples",    type=int, default=10, help="Qualitative response samples to print")
    p.add_argument("--temperature",    type=float, default=0.85)
    p.add_argument("--top_p",          type=float, default=0.9)
    return p.parse_args()


def run_evaluation(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = Tokenizer.from_file(os.path.join(args.tokenizer_dir, "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    # ── Model ────────────────────────────────────────────────────────────
    model = HappyBot(
        vocab_size=vocab_size, d_model=256, num_encoder_layers=4, num_decoder_layers=4,
        num_heads=2, d_ff=1024, dropout=0.0,
        num_emotion_classes=32, num_strategy_classes=8, pad_token_id=pad_id,
    ).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    # ── Dataset ──────────────────────────────────────────────────────────
    test_ds = HappyBotDataset(args.test_data, tokenizer, phase=2)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id=pad_id),
    )

    # ── Collect predictions ───────────────────────────────────────────────
    all_gen_losses = []
    emo_preds, emo_true, strat_preds, strat_true = [], [], [], []
    generated_seqs = []
    cross_attn_examples = []  # Store for visualization

    print(f"[Eval] Running teacher-forced evaluation on {len(test_ds)} samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            src        = batch["encoder_ids"].to(device)
            tgt_in     = batch["decoder_input_ids"].to(device)
            tgt_labels = batch["decoder_target_ids"].to(device)
            emo_labels = batch["emotion_label"].to(device)
            strat_labels = batch["strategy_label"].to(device)

            out = model(src, tgt_in)
            logits = out["logits"]

            # Generation loss (per-batch, teacher-forced)
            valid_mask = tgt_labels != -100
            if valid_mask.sum() > 0:
                flat_logits = logits[valid_mask]
                flat_labels = tgt_labels[valid_mask]
                loss = F.cross_entropy(flat_logits, flat_labels).item()
                all_gen_losses.append(loss)

            # NLU head predictions
            emo_preds.extend(out["emotion_logits"].argmax(-1).cpu().tolist())
            emo_true.extend(emo_labels.cpu().tolist())
            strat_preds.extend(out["strategy_logits"].argmax(-1).cpu().tolist())
            strat_true.extend(strat_labels.cpu().tolist())

            # Greedy generation for diversity metrics
            enc_out = model.encode(src)
            memory  = enc_out["memory"]
            src_mask = enc_out["src_mask"]
            B = src.size(0)
            tgt_cur = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
            for _ in range(args.max_gen_len):
                next_logits = model.decode_step(tgt_cur, memory, src_mask)
                next_tok = next_logits.argmax(-1, keepdim=True)
                tgt_cur = torch.cat([tgt_cur, next_tok], dim=1)
                if (next_tok == eos_id).all():
                    break
            for seq in tgt_cur.cpu().tolist():
                seq = [t for t in seq[1:] if t not in (bos_id, eos_id, pad_id)]
                generated_seqs.append(seq)

            # Save first batch cross-attention for viz
            if batch_idx == 0:
                cross_attn_examples = {
                    "cross_attn": [w[0].cpu() for w in out["cross_attn_weights"]],  # first sample
                    "encoder_attn": [w[0].cpu() for w in out["encoder_attn_weights"]],
                    "src_tokens": tokenizer.decode(src[0].cpu().tolist(), skip_special_tokens=False).split(),
                    "tgt_tokens": tokenizer.decode(
                        [t for t in tgt_in[0].cpu().tolist() if t != pad_id],
                        skip_special_tokens=False
                    ).split(),
                }

    # ── Compute metrics ───────────────────────────────────────────────────

    avg_gen_loss = sum(all_gen_losses) / max(1, len(all_gen_losses))
    perplexity = compute_perplexity(avg_gen_loss)
    d1 = compute_distinct_n(generated_seqs, 1)
    d2 = compute_distinct_n(generated_seqs, 2)

    def weighted_f1_per_class(preds, targets, ignore_idx=-1):
        valid = [(p, t) for p, t in zip(preds, targets) if t != ignore_idx]
        if not valid:
            return {}, 0.0
        ps, ts = zip(*valid)
        classes = sorted(set(ts))
        total = len(ts)
        f1_scores = {}
        weighted_sum = 0.0
        for c in classes:
            tp = sum(1 for p, t in zip(ps, ts) if p == c and t == c)
            fp = sum(1 for p, t in zip(ps, ts) if p == c and t != c)
            fn = sum(1 for p, t in zip(ps, ts) if p != c and t == c)
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            support = sum(1 for t in ts if t == c)
            f1_scores[c] = {"f1": f1, "precision": prec, "recall": rec, "support": support}
            weighted_sum += f1 * support / total
        return f1_scores, weighted_sum

    emo_f1_per_class, emo_weighted_f1 = weighted_f1_per_class(emo_preds, emo_true)
    strat_f1_per_class, strat_weighted_f1 = weighted_f1_per_class(strat_preds, strat_true, ignore_idx=-1)

    # ── Confusion matrices ────────────────────────────────────────────────
    def confusion_matrix(preds, targets, n_classes, ignore_idx=-1):
        cm = [[0] * n_classes for _ in range(n_classes)]
        for p, t in zip(preds, targets):
            if t == ignore_idx or p < 0 or p >= n_classes or t < 0 or t >= n_classes:
                continue
            cm[t][p] += 1
        return cm

    emo_cm = confusion_matrix(emo_preds, emo_true, 32)
    strat_cm = confusion_matrix(strat_preds, strat_true, 8)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Perplexity       : {perplexity:.2f}  (target < 80)")
    print(f"  Distinct-1       : {d1:.4f}  (target > 0.15)")
    print(f"  Distinct-2       : {d2:.4f}  (target > 0.40)")
    print(f"  Emotion F1 (wtd) : {emo_weighted_f1:.4f}  (target > 0.65)")
    print(f"  Strategy F1 (wtd): {strat_weighted_f1:.4f}  (target > 0.60)")
    print("=" * 60)

    print("\n  Strategy F1 per class:")
    for sid, metrics in sorted(strat_f1_per_class.items()):
        name = CANONICAL_STRATEGIES[sid] if sid < len(CANONICAL_STRATEGIES) else f"id:{sid}"
        print(f"    [{sid}] {name:<35} F1={metrics['f1']:.3f}  support={metrics['support']}")

    # ── Save results to JSON ──────────────────────────────────────────────
    results = {
        "perplexity":        perplexity,
        "distinct1":         d1,
        "distinct2":         d2,
        "emotion_f1":        emo_weighted_f1,
        "strategy_f1":       strat_weighted_f1,
        "emotion_f1_per_class": {
            str(k): v for k, v in emo_f1_per_class.items()
        },
        "strategy_f1_per_class": {
            str(k): v for k, v in strat_f1_per_class.items()
        },
    }
    results_path = os.path.join(args.output_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Eval] Metrics saved → {results_path}")

    # ── Attention visualization ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Cross-attention: layer 3 (last), head 0
        # Shows which encoder tokens the decoder attends to at each generation step
        cross_attn_layer_last = cross_attn_examples["cross_attn"][-1]  # (H, T_tgt, T_src)
        attn_head0 = cross_attn_layer_last[0].numpy()  # (T_tgt, T_src)

        src_toks = cross_attn_examples["src_tokens"][:attn_head0.shape[1]]
        tgt_toks = cross_attn_examples["tgt_tokens"][:attn_head0.shape[0]]

        fig, ax = plt.subplots(figsize=(max(8, len(src_toks) * 0.5), max(6, len(tgt_toks) * 0.4)))
        sns.heatmap(
            attn_head0[:len(tgt_toks), :len(src_toks)],
            xticklabels=src_toks,
            yticklabels=tgt_toks,
            ax=ax, cmap="Blues", annot=False,
        )
        ax.set_title("Cross-Attention Weights (Layer 4, Head 1)\nDecoder position → Encoder token")
        ax.set_xlabel("Encoder tokens (input)")
        ax.set_ylabel("Decoder positions (output)")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        cross_attn_path = os.path.join(args.output_dir, "cross_attention_heatmap.png")
        plt.savefig(cross_attn_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Eval] Cross-attention heatmap → {cross_attn_path}")

    except ImportError:
        print("[Eval] matplotlib/seaborn not installed — skipping attention visualization")

    # ── Qualitative samples ───────────────────────────────────────────────
    print(f"\n[Eval] Qualitative samples ({args.num_samples} examples):")
    print("-" * 60)

    sample_ds = HappyBotDataset(args.test_data, tokenizer, phase=2)
    indices = list(range(min(args.num_samples, len(sample_ds))))

    with torch.no_grad():
        for i in indices:
            sample = sample_ds.samples[i]
            src_text = sample["input"]
            true_response = sample["target"]
            true_emotion = sample.get("emotion_label", -1)
            true_strategy = sample.get("strategy_label", -1)

            src_enc = tokenizer.encode(src_text)
            src_ids = torch.tensor([src_enc.ids[:512]], dtype=torch.long, device=device)
            enc_out = model.encode(src_ids)
            memory  = enc_out["memory"]
            src_mask = enc_out["src_mask"]
            pred_emotion   = enc_out["emotion_logits"].argmax(-1).item()
            pred_strategy  = enc_out["strategy_logits"].argmax(-1).item()

            # Generate response with nucleus sampling
            tgt_cur = torch.tensor([[bos_id]], dtype=torch.long, device=device)
            gen_ids = []
            for _ in range(args.max_gen_len):
                logits = model.decode_step(tgt_cur, memory, src_mask)
                next_id = top_p_sampling(logits[0], temperature=args.temperature, top_p=args.top_p)
                if next_id == eos_id:
                    break
                gen_ids.append(next_id)
                tgt_cur = torch.cat([tgt_cur, torch.tensor([[next_id]], device=device)], dim=1)

            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            true_strat_name = CANONICAL_STRATEGIES[true_strategy] if 0 <= true_strategy < len(CANONICAL_STRATEGIES) else "?"
            pred_strat_name = CANONICAL_STRATEGIES[pred_strategy] if 0 <= pred_strategy < len(CANONICAL_STRATEGIES) else "?"
            true_emo_name   = EMOTION_LABELS_ED[true_emotion] if 0 <= true_emotion < len(EMOTION_LABELS_ED) else "?"
            pred_emo_name   = EMOTION_LABELS_ED[pred_emotion] if 0 <= pred_emotion < len(EMOTION_LABELS_ED) else "?"

            print(f"\n[Sample {i+1}]")
            print(f"  Input    : {src_text[:120]}...")
            print(f"  True emo : {true_emo_name}  |  Pred emo: {pred_emo_name}")
            print(f"  True strat: {true_strat_name}  |  Pred strat: {pred_strat_name}")
            print(f"  Reference: {true_response[:120]}")
            print(f"  Generated: {gen_text[:120]}")

    print("\n[Eval] Evaluation complete.")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)