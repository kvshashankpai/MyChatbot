"""
scripts/evaluate.py — Full Evaluation Suite

Implements all evaluation tasks from Section 8:
  1. NLU head evaluation: emotion F1, strategy F1, confusion matrices
  2. NLG evaluation: perplexity, Distinct-1/2, BLEU-4
  3. Attention visualization: encoder CLS attention heatmaps,
     decoder cross-attention heatmaps

Run command:
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
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer import (
    HappyBotTokenizer,
    ID_TO_STRATEGY,
    NUM_STRATEGY_CLASSES,
    NUM_EMOTION_CLASSES_PHASE2,
)
from dataset import HappyBotDataset, build_dataloader
from model.transformer import HappyBot
from inference import HappyBotInference
from utils import (
    load_checkpoint,
    compute_distinct_n,
    compute_perplexity,
    compute_f1_score,
    build_generation_loss,
    get_logger,
)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    predictions: List[int],
    targets: List[int],
    class_names: List[str],
    title: str,
    output_path: str,
) -> None:
    """Plot and save confusion matrix using seaborn."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=ax
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True",      fontsize=12)
        ax.set_title(title,        fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Confusion matrix saved: {output_path}")

    except ImportError:
        print("matplotlib/seaborn/sklearn not installed. Skipping confusion matrix plot.")


# ─────────────────────────────────────────────────────────────────────────────
# Attention heatmap visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_heatmap(
    attn_weights: torch.Tensor,   # (num_heads, T_q, T_k)
    x_labels: List[str],
    y_labels: List[str],
    title: str,
    output_path: str,
    max_tokens: int = 30,
) -> None:
    """
    Plot attention weight heatmap.

    For encoder CLS attention: T_q = 1 (CLS queries all tokens),
    T_k = sequence length.

    For cross-attention: T_q = decoder steps, T_k = encoder sequence.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        # Average over heads for clarity; shape → (T_q, T_k)
        avg_attn = attn_weights.mean(dim=0).cpu().float().numpy()

        # Truncate for display
        avg_attn = avg_attn[:max_tokens, :max_tokens]
        x_labels = x_labels[:max_tokens]
        y_labels = y_labels[:max_tokens]

        fig, ax = plt.subplots(figsize=(min(20, len(x_labels) * 0.5 + 2),
                                         min(12, len(y_labels) * 0.5 + 2)))
        sns.heatmap(
            avg_attn, xticklabels=x_labels, yticklabels=y_labels,
            cmap="viridis", ax=ax
        )
        ax.set_title(title, fontsize=13)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Attention heatmap saved: {output_path}")

    except ImportError:
        print("matplotlib/seaborn not installed. Skipping attention visualization.")


# ─────────────────────────────────────────────────────────────────────────────
# BLEU-4 computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu4(hypotheses: List[str], references: List[str]) -> float:
    """Compute corpus-level BLEU-4 score."""
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        hyps = [h.lower().split() for h in hypotheses]
        refs = [[r.lower().split()] for r in references]
        return corpus_bleu(refs, hyps, smoothing_function=smooth)
    except ImportError:
        print("nltk not installed. Skipping BLEU. pip install nltk")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation routine
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(args):
    logger  = get_logger("evaluate")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load tokenizer and model ───────────────────────────────────────────
    tokenizer = HappyBotTokenizer(args.tokenizer_dir)
    model = HappyBot(
        vocab_size=tokenizer.vocab_size,
        d_model=256, num_heads=2,
        num_encoder_layers=4, num_decoder_layers=4,
        d_ff=1024, dropout=0.0,
        num_emotion_classes=NUM_EMOTION_CLASSES_PHASE2,
        num_strategy_classes=NUM_STRATEGY_CLASSES,
        pad_token_id=tokenizer.pad_id,
    ).to(device)
    load_checkpoint(args.checkpoint, model, device=device)
    model.eval()

    # ── DataLoader ─────────────────────────────────────────────────────────
    test_ds = HappyBotDataset(
        args.test_data, tokenizer, max_src_len=512, max_tgt_len=128, phase=2
    )
    test_loader = build_dataloader(test_ds, batch_size=16, pad_id=tokenizer.pad_id,
                                   shuffle=False, num_workers=0)

    gen_loss_fn = build_generation_loss(tokenizer.vocab_size, tokenizer.pad_id)

    # ── Evaluation loop ────────────────────────────────────────────────────
    total_gen_loss = 0.0
    emotion_preds,  emotion_tgts   = [], []
    strategy_preds, strategy_tgts  = [], []
    hypotheses, references = [], []
    n_batches = 0

    logger.info("Running evaluation loop...")
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            encoder_ids=batch["encoder_ids"],
            decoder_input=batch["decoder_input"],
            encoder_mask=batch["encoder_mask"],
            decoder_mask=batch["decoder_mask"],
        )

        # Generation loss
        B, T, V = out["gen_logits"].shape
        total_gen_loss += gen_loss_fn(out["gen_logits"], batch["decoder_target"]).item()

        # NLU predictions
        emotion_preds.extend(out["emotion_logits"].argmax(-1).cpu().tolist())
        emotion_tgts.extend(batch["emotion_label"].cpu().tolist())
        strategy_preds.extend(out["strategy_logits"].argmax(-1).cpu().tolist())
        strategy_tgts.extend(batch["strategy_label"].cpu().tolist())

        # Greedy decode for BLEU / Distinct
        greedy_ids = out["gen_logits"].argmax(-1).cpu()
        tgt_ids    = batch["decoder_target"].cpu()
        for i in range(B):
            hyp = tokenizer.decode(greedy_ids[i].tolist(), skip_special_tokens=True)
            ref = tokenizer.decode(
                [t for t in tgt_ids[i].tolist() if t != -100],
                skip_special_tokens=True
            )
            hypotheses.append(hyp)
            references.append(ref)

        n_batches += 1

    # ── Compute metrics ───────────────────────────────────────────────────
    avg_gen_loss = total_gen_loss / n_batches
    perplexity   = compute_perplexity(avg_gen_loss)
    distinct1    = compute_distinct_n(hypotheses, n=1)
    distinct2    = compute_distinct_n(hypotheses, n=2)
    bleu4        = compute_bleu4(hypotheses, references)

    valid_emo_preds = [p for p, t in zip(emotion_preds, emotion_tgts) if t >= 0]
    valid_emo_tgts  = [t for t in emotion_tgts if t >= 0]
    valid_str_preds = [p for p, t in zip(strategy_preds, strategy_tgts) if t >= 0]
    valid_str_tgts  = [t for t in strategy_tgts if t >= 0]

    emotion_f1  = compute_f1_score(valid_emo_preds,  valid_emo_tgts,
                                   num_classes=NUM_EMOTION_CLASSES_PHASE2)
    strategy_f1 = compute_f1_score(valid_str_preds, valid_str_tgts,
                                   num_classes=NUM_STRATEGY_CLASSES)

    metrics = {
        "perplexity":   perplexity,
        "distinct1":    distinct1,
        "distinct2":    distinct2,
        "bleu4":        bleu4,
        "emotion_f1":   emotion_f1,
        "strategy_f1":  strategy_f1,
    }

    logger.info("=" * 55)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 55)
    for k, v in metrics.items():
        target = {
            "perplexity": "< 80",
            "distinct1":  "> 0.15",
            "distinct2":  "> 0.40",
            "emotion_f1": "> 0.65",
            "strategy_f1":"> 0.60",
        }.get(k, "")
        logger.info(f"  {k:<16} {v:.4f}   (target: {target})")
    logger.info("=" * 55)

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Confusion matrices ─────────────────────────────────────────────────
    strategy_names = [ID_TO_STRATEGY.get(i, str(i)) for i in range(NUM_STRATEGY_CLASSES)]
    emotion_names  = [
        "anxiety", "sadness", "anger", "fear",
        "disgust", "joy",     "surprise", "neutral"
    ][:NUM_EMOTION_CLASSES_PHASE2]

    plot_confusion_matrix(
        valid_str_preds, valid_str_tgts, strategy_names,
        "Strategy Head Confusion Matrix",
        os.path.join(args.output_dir, "strategy_confusion.png")
    )
    plot_confusion_matrix(
        valid_emo_preds, valid_emo_tgts, emotion_names,
        "Emotion Head Confusion Matrix",
        os.path.join(args.output_dir, "emotion_confusion.png")
    )

    # ── Attention visualization on held-out examples ───────────────────────
    if args.viz_examples:
        engine = HappyBotInference(model, tokenizer, device)
        for i, example_input in enumerate(args.viz_examples.split("||")):
            example_input = example_input.strip()
            result = engine.generate_with_attention(example_input)
            strategy = result["strategy"]
            enc_tokens = result.get("encoder_tokens", [])

            # Cross-attention: shape (num_layers, num_heads, T_q, T_k)
            for layer_idx, cross_attn in enumerate(result.get("cross_attn", [])):
                # cross_attn: (1, num_heads, T_decoder, T_encoder)
                ca = cross_attn.squeeze(0)  # (num_heads, T_dec, T_enc)
                resp_tokens = result["response"].split()[:ca.size(1)]
                plot_attention_heatmap(
                    ca,
                    x_labels=enc_tokens,
                    y_labels=resp_tokens,
                    title=f"Cross-Attn Layer {layer_idx} | Strategy: {strategy}",
                    output_path=os.path.join(
                        args.output_dir,
                        f"cross_attn_ex{i}_layer{layer_idx}.png"
                    ),
                )

    logger.info(f"All results saved to {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--tokenizer_dir", default="data/tokenizer")
    p.add_argument("--test_data",     default="data/processed/esconv_test.jsonl")
    p.add_argument("--output_dir",    default="outputs/evaluation")
    p.add_argument("--viz_examples",  default=None,
                   help="Pipe-delimited (||) example sentences for attention viz")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
