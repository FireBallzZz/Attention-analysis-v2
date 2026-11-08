"""Manual-Tuned Binary Attention Classifier.

Canonical architecture for the active classroom attention classifier:
- Backbone: AttentionFusionNet (frame_encoder + bidirectional GRU)
- Frame Encoder: Frozen pre-trained linear projection
- Temporal Pooling: Concatenation of [Last-Step (128) + Mean-Pool (128) + Max-Pool (128)] = 384-D
- Classification Head: 2-layer MLP with BatchNorm, GELU, and Dropout
- Output: 2 classes: Index 0 = LOW (Non-Attentive), Index 1 = HIGH (Attentive)
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from attention_pipeline.models.attention_fusion import AttentionFusionNet


class ManualBinaryAttentionClassifier(nn.Module):
    def __init__(
        self,
        base_net: AttentionFusionNet,
        freeze_frame_encoder: bool = True,
        dropout_1: float = 0.2,
        dropout_2: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_encoder = base_net.frame_encoder
        self.temporal = base_net.temporal
        self.freeze_frame_encoder = freeze_frame_encoder

        if freeze_frame_encoder:
            for p in self.frame_encoder.parameters():
                p.requires_grad = False

        # Concat last (128) + mean (128) + max (128) = 384-D
        self.head = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout_2),
            nn.Linear(64, 2),
        )

    def encode_temporal(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if self.freeze_frame_encoder:
            with torch.no_grad():
                enc = self.frame_encoder(x.reshape(b * t, d)).reshape(b, t, -1)
        else:
            enc = self.frame_encoder(x.reshape(b * t, d)).reshape(b, t, -1)
        out, _ = self.temporal(enc)  # (B, T, 128)
        last = out[:, -1, :]
        mean = torch.mean(out, dim=1)
        max_v, _ = torch.max(out, dim=1)
        return torch.cat([last, mean, max_v], dim=-1)  # (B, 384)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.encode_temporal(x)
        return self.head(fused)

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @torch.inference_mode()
    def score(self, window: torch.Tensor) -> torch.Tensor:
        """Returns attention score in [0.0, 100.0] for pipeline compatibility.
        
        Score represents P(HIGH) * 100.0.
        Score >= 50.0 corresponds to HIGH (Attentive).
        Score < 50.0 corresponds to LOW (Non-Attentive).
        """
        if window.dim() == 2:
            window = window.unsqueeze(0)
        probs = self.predict_proba(window)
        return probs[:, 1] * 100.0

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> "ManualBinaryAttentionClassifier":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_state_dict(state)
        self.eval()
        return self
