"""
encoder.py — Transformer Encoder stack.
Includes:
  - Sinusoidal Positional Encoding (pre-computed, no parameters)
  - EncoderLayer (MHA + FFN + residuals + LayerNorm)
  - Encoder (embedding + PE + stack + CLS-based NLU heads)
  - EmotionHead and StrategyHead (2-layer MLPs on the [CLS] token)
"""

import math
import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).
    Registered as a buffer — no gradients, moves with the model to GPU.
    Extrapolates cleanly to unseen sequence lengths.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) — add positional encoding in-place"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Position-wise FFN: Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model).
    Standard 4× expansion: d_ff = 4 * d_model = 1024.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """
    Single encoder layer:
      x → MHA(x,x,x) → add+norm → FFN → add+norm
    Pre-norm variant: LayerNorm BEFORE the sublayer (more stable for small
    from-scratch training). This is the key fix over the naive post-norm.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # (B, T, d_model)
        src_mask: torch.Tensor,    # (B, 1, 1, T) — padding mask, True=attend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm self-attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.self_attn(x_norm, x_norm, x_norm, mask=src_mask)
        x = residual + self.dropout(attn_out)

        # Pre-norm FFN
        residual = x
        x = residual + self.dropout(self.ffn(self.norm2(x)))

        return x, attn_weights


class EmotionHead(nn.Module):
    """
    MLP classification head for emotion detection.
    Operates on the [CLS] token (position 0) from the final encoder layer.
    """

    def __init__(self, d_model: int, num_emotion_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotion_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, cls_vector: torch.Tensor) -> torch.Tensor:
        """cls_vector: (B, d_model) → logits: (B, num_emotion_classes)"""
        return self.net(cls_vector)


class StrategyHead(nn.Module):
    """
    MLP classification head for therapeutic strategy prediction.
    Operates on the same [CLS] token.
    8 strategy classes matching ESConv canonical labels.
    """

    def __init__(self, d_model: int, num_strategy_classes: int = 8, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_strategy_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, cls_vector: torch.Tensor) -> torch.Tensor:
        """cls_vector: (B, d_model) → logits: (B, num_strategy_classes)"""
        return self.net(cls_vector)


class Encoder(nn.Module):
    """
    Full Encoder stack:
      token_embedding + sinusoidal_PE → N × EncoderLayer → EmotionHead + StrategyHead

    IMPORTANT: embedding is passed in from HappyBot (shared with decoder input embedding).
    The encoder does NOT own the embedding — it receives it from the top-level module.
    This enforces weight tying.
    """

    def __init__(
        self,
        embedding: nn.Embedding,       # shared embedding from HappyBot
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 2,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        num_emotion_classes: int = 32,   # EmpatheticDialogues has 32 emotion labels
        num_strategy_classes: int = 8,   # ESConv canonical strategies
    ):
        super().__init__()
        self.embedding = embedding
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # final pre-norm after all layers

        self.emotion_head = EmotionHead(d_model, num_emotion_classes, dropout)
        self.strategy_head = StrategyHead(d_model, num_strategy_classes, dropout)

    def forward(
        self,
        src: torch.Tensor,       # (B, T_src)  token ids
        src_mask: torch.Tensor,  # (B, 1, 1, T_src)  True=attend (not padding)
    ) -> dict:
        """
        Returns a dict:
          "memory": (B, T_src, d_model) — encoder output passed to cross-attention
          "emotion_logits": (B, num_emotion_classes)
          "strategy_logits": (B, num_strategy_classes)
          "attn_weights": list of (B, H, T, T) per layer (for visualization)
        """
        # Embedding + scale + positional encoding
        x = self.embedding(src) * self.scale    # (B, T, d_model)
        x = self.pos_encoding(x)

        attn_weights_all = []
        for layer in self.layers:
            x, attn_w = layer(x, src_mask)
            attn_weights_all.append(attn_w)

        x = self.norm(x)  # Final LayerNorm

        # CLS token is always at position 0
        cls_vector = x[:, 0, :]  # (B, d_model)

        emotion_logits = self.emotion_head(cls_vector)    # (B, num_emotion_classes)
        strategy_logits = self.strategy_head(cls_vector)  # (B, num_strategy_classes)

        return {
            "memory": x,
            "emotion_logits": emotion_logits,
            "strategy_logits": strategy_logits,
            "cls_vector": cls_vector,
            "attn_weights": attn_weights_all,
        }