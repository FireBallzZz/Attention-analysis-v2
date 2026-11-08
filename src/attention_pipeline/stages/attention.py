"""Stage 4: windowed AttentionFusionNet scoring."""

from __future__ import annotations

import numpy as np
import torch

from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.rubric import RubricResult, classify_attention, classify_from_features
from attention_pipeline.types import StudentFrame, TrackState


class AttentionScorer:
    def __init__(
        self,
        model: AttentionFusionNet | None,
        device: torch.device | None,
        window: int = 16,
        max_gap: float = 0.5,
    ) -> None:
        self.model = model
        self.device = device
        self.window = window
        self.max_gap = max_gap
        self.use_heuristic = False

    def push(
        self,
        track: TrackState,
        feat: np.ndarray,
        timestamp: float,
        obs: StudentFrame | None = None,
        lesson_yaw: float | None = None,
    ) -> tuple[float | None, RubricResult, float | None]:
        ts = float(timestamp)
        # Check temporal continuity: if gap > max_gap or timestamps reset, break window
        if track.timestamps:
            prev_ts = track.timestamps[-1]
            if (ts - prev_ts > self.max_gap) or (ts < prev_ts):
                track.feature_window.clear()
                track.timestamps.clear()
                track.scores.clear()
                if hasattr(track, "fusion_scores"):
                    track.fusion_scores.clear()

        track.feature_window.append(feat.astype(np.float32))
        track.timestamps.append(ts)
        if len(track.feature_window) > self.window:
            track.feature_window = track.feature_window[-self.window :]
            track.timestamps = track.timestamps[-self.window :]
            track.scores = track.scores[-(self.window - 1) :]
            if hasattr(track, "fusion_scores") and len(track.fusion_scores) >= self.window:
                track.fusion_scores = track.fusion_scores[-(self.window - 1) :]

        rubric = (
            classify_attention(obs, lesson_yaw=lesson_yaw)
            if obs is not None
            else classify_from_features(feat, lesson_yaw=lesson_yaw)
        )
        fusion_score: float | None = None
        if not self.use_heuristic and self.model is not None and self.device is not None:
            if len(track.feature_window) >= self.window:
                stacked = np.stack(track.feature_window[-self.window :], axis=0)
                x = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
                fusion_score = float(self.model.score(x).item())
                score = fusion_score
            else:
                score = None
        else:
            score = rubric.score

        track.scores.append(score)
        if hasattr(track, "fusion_scores"):
            track.fusion_scores.append(fusion_score)
        return score, rubric, fusion_score
