"""Native 4-Class Student Engagement Classifier.

Architecture for native multiclass engagement classification:
- Backbone: AttentionFusionNet (frame_encoder + bidirectional GRU)
- Frame Encoder: Frozen pre-trained linear projection
- Temporal Pooling: Concatenation of [Last-Step (128) + Mean-Pool (128) + Max-Pool (128)] = 384-D
- Classification Head: 2-layer MLP with BatchNorm, GELU, and Dropout
- Output: 4 classes (Native discrete engagement levels):
    Index 0: Very Low Engagement (DAiSEE Level 0)
    Index 1: Low Engagement      (DAiSEE Level 1)
    Index 2: High Engagement     (DAiSEE Level 2)
    Index 3: Very High Engagement (DAiSEE Level 3)
"""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from attention_pipeline.models.attention_fusion import AttentionFusionNet


ENGAGEMENT_CLASS_NAMES: dict[int, str] = {
    0: "Very Low Engagement",
    1: "Low Engagement",
    2: "High Engagement",
    3: "Very High Engagement",
}


class Native4ClassEngagementClassifier(nn.Module):
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
            nn.Linear(64, 4),  # Native 4-class output
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
        """Returns softmax probabilities of shape (batch_size, 4)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns discrete class IDs in {0, 1, 2, 3} via argmax."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    @torch.inference_mode()
    def predict_with_metadata(self, window: torch.Tensor) -> dict[str, Any]:
        """Provides full 4-class output for pipeline integration.
        
        Returns:
            predicted_class: int in {0, 1, 2, 3}
            class_name: human-readable engagement string
            probabilities: list of 4 float probabilities
        """
        if window.dim() == 2:
            window = window.unsqueeze(0)
        probs = self.predict_proba(window)[0].cpu().tolist()
        pred_cls = int(torch.argmax(torch.tensor(probs)).item())
        return {
            "predicted_class": pred_cls,
            "class_name": ENGAGEMENT_CLASS_NAMES.get(pred_cls, f"Class_{pred_cls}"),
            "probabilities": [round(p, 4) for p in probs],
            "confidence": round(float(probs[pred_cls]), 4),
        }

    def load(self, path: str | Path, map_location: str | torch.device = "cpu") -> "Native4ClassEngagementClassifier":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.load_state_dict(state)
        self.eval()
        return self
