"""Independent Inference Pipeline for Native 4-Class Student Engagement Classification.

Provides end-to-end inference over temporal feature windows without using binary logic
or binary HIGH/LOW overlays.

Class Contract:
- Class 0: Very Low Engagement
- Class 1: Low Engagement
- Class 2: High Engagement
- Class 3: Very High Engagement
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import deque
import numpy as np
import torch

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.native_multiclass import (
    Native4ClassEngagementClassifier,
    ENGAGEMENT_CLASS_NAMES,
)


class Engagement4ClassPipeline:
    def __init__(
        self,
        checkpoint_path: str | Path,
        backbone_path: str | Path | None = None,
        device: str | torch.device = "cuda:0" if torch.cuda.is_available() else "cpu",
        temporal_window: int = 16,
        enable_temporal_smoothing: bool = False,
        smoothing_window: int = 5,
    ) -> None:
        self.device = torch.device(device)
        self.temporal_window = temporal_window
        self.enable_temporal_smoothing = enable_temporal_smoothing
        self.smoothing_window = smoothing_window

        # Initialize backbone
        self.base_net = AttentionFusionNet(feat_dim=32, hidden=64)
        if backbone_path and Path(backbone_path).exists():
            self.base_net.load(backbone_path)

        # Initialize native 4-class classifier
        self.model = Native4ClassEngagementClassifier(self.base_net, freeze_frame_encoder=True).to(self.device)
        self.model.load(checkpoint_path, map_location=self.device)
        self.model.eval()

        # Track history buffers for temporal smoothing: track_id -> deque of prob vectors
        self._history_buffers: dict[int, deque[np.ndarray]] = {}

    def predict_window(self, window_tensor: torch.Tensor, track_id: int | None = None) -> dict[str, Any]:
        """Runs forward inference on a single (16, 32) temporal feature window.
        
        Args:
            window_tensor: Tensor of shape (16, 32) or (1, 16, 32).
            track_id: Optional tracking identifier for session smoothing.
            
        Returns:
            Dictionary with predicted class, label name, probabilities, and confidence.
        """
        if window_tensor.dim() == 2:
            window_tensor = window_tensor.unsqueeze(0)
        window_tensor = window_tensor.to(self.device)

        with torch.no_grad():
            raw_probs = self.model.predict_proba(window_tensor)[0].cpu().numpy()

        # Optional temporal smoothing
        if self.enable_temporal_smoothing and track_id is not None:
            if track_id not in self._history_buffers:
                self._history_buffers[track_id] = deque(maxlen=self.smoothing_window)
            self._history_buffers[track_id].append(raw_probs)
            effective_probs = np.mean(list(self._history_buffers[track_id]), axis=0)
        else:
            effective_probs = raw_probs

        pred_class_id = int(np.argmax(effective_probs))
        class_name = ENGAGEMENT_CLASS_NAMES.get(pred_class_id, f"Class_{pred_class_id}")
        confidence = float(effective_probs[pred_class_id])

        return {
            "predicted_class": pred_class_id,
            "class_name": class_name,
            "probabilities": [round(float(p), 4) for p in effective_probs],
            "raw_probabilities": [round(float(p), 4) for p in raw_probs],
            "confidence": round(confidence, 4),
            "smoothing_applied": self.enable_temporal_smoothing,
            "track_id_note": "Track ID represents an anonymous visual bounding-box track, not verified physical identity."
        }

    def reset_track(self, track_id: int) -> None:
        """Clears temporal smoothing history for a specific track ID."""
        if track_id in self._history_buffers:
            del self._history_buffers[track_id]
