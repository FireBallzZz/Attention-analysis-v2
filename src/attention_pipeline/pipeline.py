"""End-to-end runner for the six-stage attention pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from attention_pipeline.features import to_student_frame
from attention_pipeline.rubric import apply_peer_cues, classify_attention
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.drop_detector import AttentionDropDetector, DropConfig
from attention_pipeline.models.l2cs_net import L2CSNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier
from attention_pipeline.models.posture_mlp import PostureMLP
from attention_pipeline.stages.attention import AttentionScorer
from attention_pipeline.stages.detect_track import DetectorTracker
from attention_pipeline.stages.intervention import recommend
from attention_pipeline.stages.modalities import FaceModalityExtractor, PostureExtractor
from attention_pipeline.stages.scb_behavior import SCBBehaviorDetector
from attention_pipeline.stages.source import iter_frames
from attention_pipeline.types import POSTURE_LABELS, TrackState
from attention_pipeline.utils import load_yaml, project_path, resolve_device


class AttentionPipeline:
    def __init__(self, cfg: dict[str, Any], load_weights: bool = True) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.get("device", "auto"))
        models_cfg = cfg.get("models", {})
        data_cfg = cfg.get("data", {})

        self.gaze = L2CSNet().to(self.device).eval()
        self.posture = PostureMLP().to(self.device).eval()
        fusion_cfg = cfg.get("fusion", {})
        self.fusion_base = AttentionFusionNet(
            feat_dim=fusion_cfg.get("feat_dim", 32),
            hidden=fusion_cfg.get("hidden", 64),
        ).to(self.device).eval()
        self.fusion = ManualBinaryAttentionClassifier(self.fusion_base).to(self.device).eval()

        loaded_fusion = False
        loaded_gaze = loaded_posture = False
        if load_weights:
            loaded_gaze = self._maybe_load(self.gaze, models_cfg.get("gaze_weights"))
            loaded_posture = self._maybe_load(self.posture, models_cfg.get("posture_weights"))
            binary_weights = models_cfg.get("manual_binary_weights") or models_cfg.get("fusion_weights")
            loaded_fusion = self._maybe_load(self.fusion, binary_weights)
            if not loaded_fusion:
                self.fusion = self.fusion_base
                loaded_fusion = self._maybe_load(self.fusion, models_cfg.get("fusion_weights"))

        detect_cfg = cfg.get("detect", {})
        track_cfg = cfg.get("track", {})
        self.detector = DetectorTracker(
            weights=detect_cfg.get("weights", "yolo11n-pose.pt"),
            person_weights=detect_cfg.get("person_weights", "yolo11n.pt"),
            conf=detect_cfg.get("conf", 0.08),
            iou=detect_cfg.get("iou", 0.45),
            device=str(self.device),
            max_age=track_cfg.get("max_age", 45),
            n_init=track_cfg.get("n_init", 1),
            imgsz=int(detect_cfg.get("imgsz", 960)),
            tiles=bool(detect_cfg.get("tiles", True)),
            min_det_conf=float(detect_cfg.get("min_det_conf", 0.25)),
            require_face=bool(detect_cfg.get("require_face", True)),
            max_people=detect_cfg.get("max_people") or detect_cfg.get("max_students"),
            smooth_box=bool(track_cfg.get("smooth_box", True)),
            box_ema_alpha=float(track_cfg.get("box_ema_alpha", 0.7)),
            max_area_ratio=float(track_cfg.get("max_area_ratio", 1.5)),
            auto_max_students=bool(track_cfg.get("auto_max_students", True)),
        )
        face_cfg = cfg.get("face", {})
        self.faces = FaceModalityExtractor(
            gaze=self.gaze,
            device=self.device,
            backend=face_cfg.get("backend", "stub"),
            min_face_px=face_cfg.get("min_face_px", 32),
            use_l2cs=loaded_gaze,
        )
        self.posture_x = PostureExtractor(self.posture, self.device, use_learned=loaded_posture)
        scb_weights = project_path(models_cfg.get("scb_weights"))
        self.scb = SCBBehaviorDetector(
            weights=scb_weights,
            device=str(self.device),
            conf=float(cfg.get("scb", {}).get("conf", 0.25)),
        )
        self.scb_data_yaml = project_path(data_cfg.get("scb_yaml"))
        self.scorer = AttentionScorer(
            self.fusion,
            self.device,
            window=fusion_cfg.get("window", 16),
            max_gap=float(fusion_cfg.get("max_gap", 0.5)),
        )
        self.scorer.use_heuristic = not loaded_fusion
        self.binary_threshold = float(fusion_cfg.get("binary_threshold", 0.50))
        drop_cfg = cfg.get("drop", {})
        self.drop = AttentionDropDetector(DropConfig(**drop_cfg) if drop_cfg else DropConfig())
        phone_w = detect_cfg.get("person_weights") or "yolo11n.pt"
        try:
            from ultralytics import YOLO
            self.phone_model = YOLO(phone_w)
        except Exception:
            self.phone_model = None
        self.tracks: dict[int, TrackState] = {}
        self._prev_crops: dict[int, tuple[float, np.ndarray]] = {}
        self._phone_boxes_recent: list[tuple[int, float, float]] = []
        self._frame_count: int = 0
        self._lesson_yaw: float | None = None
        self.dataset_status = {
            "l2cs_gaze": loaded_gaze,
            "scb_behavior": self.scb.enabled,
            "scb_yaml": bool(self.scb_data_yaml and self.scb_data_yaml.exists()),
            "fusion": loaded_fusion,
        }

    @classmethod
    def from_yaml(cls, path: str | Path, load_weights: bool = True) -> "AttentionPipeline":
        return cls(load_yaml(str(path)), load_weights=load_weights)

    def _maybe_load(self, module, path: str | None) -> bool:
        p = project_path(path)
        if p is None or not p.exists():
            return False
        module.load(p, map_location=self.device)
        return True

    def process_frame(self, timestamp: float, frame: np.ndarray) -> dict[str, Any]:
        self._frame_count += 1
        detections = self.detector.infer(frame)
        scb_dets = self.scb.infer(frame)
        scb_cfg = self.cfg.get("scb", {})
        min_iou = float(scb_cfg.get("match_iou", 0.4))
        blend = float(scb_cfg.get("blend", 0.65))
        student_boxes = [det.bbox_xyxy for det in detections]
        scb_matches = self.scb.match_all(student_boxes, scb_dets, min_iou=min_iou)

        # Prune phone detections older than 30 frames (3.0s)
        self._phone_boxes_recent = [
            (f_idx, px, py)
            for f_idx, px, py in self._phone_boxes_recent
            if self._frame_count - f_idx <= 30
        ]
        if getattr(self, "phone_model", None) is not None:
            try:
                p_res = self.phone_model.predict(
                    frame, conf=0.18, imgsz=1280, classes=[67], verbose=False, device=self.device
                )[0]
                for box in p_res.boxes:
                    bx = box.xyxy[0].cpu().numpy()
                    self._phone_boxes_recent.append(
                        (self._frame_count, float((bx[0] + bx[2]) / 2.0), float((bx[1] + bx[3]) / 2.0))
                    )
            except Exception:
                pass

        if len(self._prev_crops) > 50:
            stale = [
                tid for tid, (ts, _) in self._prev_crops.items()
                if (timestamp - ts > 5.0) or (timestamp < ts)
            ]
            for tid in stale:
                del self._prev_crops[tid]
            if len(self._prev_crops) > 100:
                oldest_tids = sorted(self._prev_crops.keys(), key=lambda k: self._prev_crops[k][0])
                for tid in oldest_tids[: len(self._prev_crops) - 50]:
                    del self._prev_crops[tid]

        observed: list[tuple[TrackState, Any]] = []
        for det, scb_probs in zip(detections, scb_matches):
            track = self.tracks.setdefault(det.track_id, TrackState(student_id=det.track_id))
            track.last_bbox = det.bbox_xyxy
            face = self.faces.extract(frame, det, det.track_id)

            # Match phone box to student desk/lap zone with spatial memory
            sx1, sy1, sx2, sy2 = det.bbox_xyxy
            sbw = sx2 - sx1
            sbh = sy2 - sy1
            has_phone_obj = False
            for _, px, py in self._phone_boxes_recent:
                if sx1 <= px <= sx2 and (sy1 + 0.35 * sbh) <= py <= (sy2 + 0.15 * sbh):
                    has_phone_obj = True
                    break

            # Direct crop-level high-resolution phone verification for seated desk/lap zone
            if getattr(self, "phone_model", None) is not None and not has_phone_obj:
                cy1 = max(0, int(sy1 + 0.35 * sbh))
                cy2 = min(frame.shape[0], int(sy2 + 0.15 * sbh))
                cx1 = max(0, int(sx1))
                cx2 = min(frame.shape[1], int(sx2))
                if cy2 > cy1 and cx2 > cx1:
                    desk_crop = frame[cy1:cy2, cx1:cx2]
                    if desk_crop.size > 0:
                        try:
                            c_res = self.phone_model.predict(
                                desk_crop, conf=0.18, classes=[67], verbose=False, device=self.device
                            )[0]
                            if c_res.boxes is not None and len(c_res.boxes) > 0:
                                has_phone_obj = True
                        except Exception:
                            pass

            # Feature 18: Crop-level optical motion matching extract_classroom_sequences.py
            x1_i, y1_i = max(0, int(det.bbox_xyxy[0])), max(0, int(det.bbox_xyxy[1]))
            x2_i, y2_i = min(frame.shape[1], int(det.bbox_xyxy[2])), min(frame.shape[0], int(det.bbox_xyxy[3]))
            crop = frame[y1_i:y2_i, x1_i:x2_i]
            motion = 0.0
            if det.track_id in self._prev_crops and crop.size > 0:
                prev_ts, prev_crop = self._prev_crops[det.track_id]
                # Require identical shape and continuous temporal arrival (<= 0.5s gap)
                if prev_crop.shape == crop.shape and 0.0 <= (timestamp - prev_ts) <= 0.5:
                    diff = cv2.absdiff(crop, prev_crop)
                    motion = float(np.mean(diff) / 255.0)
            if crop.size > 0:
                self._prev_crops[det.track_id] = (timestamp, crop.copy())

            posture = self.posture_x.predict(det.keypoints, det.bbox_xyxy)
            if scb_probs is not None:
                # Blend SCB with geometric pose so one wrong SCB box cannot dominate.
                posture = (blend * scb_probs + (1.0 - blend) * posture).astype(np.float32)
                posture = posture / max(float(posture.sum()), 1e-6)
            obs = to_student_frame(
                student_id=det.track_id,
                timestamp=timestamp,
                bbox=det.bbox_xyxy,
                keypoints=det.keypoints,
                head_pose=face.head_pose,
                gaze=face.gaze,
                emotion_probs=face.emotion_probs,
                posture_probs=posture,
                motion=motion,
                eye_aspect=face.eye_aspect,
                mouth_aspect=face.mouth_aspect,
                has_phone_obj=has_phone_obj,
            )
            observed.append((track, obs))

        if observed:
            med = float(np.median([obs.head_pose[0] for _, obs in observed]))
            if self._lesson_yaw is None:
                self._lesson_yaw = med
            else:
                self._lesson_yaw = 0.2 * med + 0.8 * self._lesson_yaw

        per_student = []
        events = []
        pending = []
        for track, obs in observed:
            score, rubric, fusion_score = self.scorer.push(
                track, obs.features, timestamp, obs=obs, lesson_yaw=self._lesson_yaw
            )
            pending.append((track, obs, score, rubric, fusion_score))

        peer = apply_peer_cues(
            [obs for _track, obs, _score, _rubric, _fs in pending],
            [rubric for _track, _obs, _score, rubric, _fs in pending],
            self._lesson_yaw,
        )
        for (track, obs, score, _old, fusion_score), rubric in zip(pending, peer):
            # Multimodal Action-Aware Neural Attention Scoring:
            # If verified active academic behavior (reading/writing/hand-raising) is detected by the rubric,
            # head pitch down corresponds to productive desk study, not disengagement.
            if fusion_score is not None:
                h_score = float(rubric.score)
                cue_str = str(rubric.cue).lower()
                is_active_study = (
                    h_score >= 70.0
                    and any(w in cue_str for w in ("reading", "writing", "hand", "watching", "presenting", "oriented"))
                    and not any(w in cue_str for w in ("phone", "sleep", "table", "left", "right", "talk", "disturb", "away"))
                )
                is_disengaged = (
                    rubric.band == "LOW"
                    or h_score <= 40.0
                    or any(w in cue_str for w in ("phone", "sleep", "table", "talk", "disturb", "left", "right", "away"))
                )
                if is_disengaged:
                    # Grounded disengagement penalty: enforce LOW attention for phone, peer talking, sleeping, looking away
                    effective_score = min(float(fusion_score), h_score)
                elif is_active_study:
                    # Blend temporal neural score with verified active study action prior
                    effective_score = float(0.4 * fusion_score + 0.6 * h_score)
                else:
                    effective_score = float(fusion_score)

                band = "HIGH" if (effective_score / 100.0) >= self.binary_threshold else "LOW"
                cue = rubric.cue
            elif not self.scorer.use_heuristic:
                # In neural mode: student is in warmup until 16 sequential observations are available
                effective_score = None
                band = "WARMUP"
                cue = "warmup"
            else:
                effective_score = float(rubric.score)
                band, cue = rubric.band, rubric.cue
            event = self.drop.update(track, obs.features) if effective_score is not None else None
            rec = recommend(event, class_mean_score=_mean_score(self.tracks), cue=cue) if event else None
            if event:
                events.append({"event": event, "recommendation": rec})
            per_student.append(
                {
                    "student_id": obs.student_id,
                    "bbox": obs.bbox.tolist(),
                    "score": effective_score,
                    "fusion_score": float(fusion_score) if fusion_score is not None else None,
                    "heuristic_score": float(rubric.score),
                    "band": band,
                    "cue": cue,
                    "head_pose": obs.head_pose.tolist(),
                    "gaze": obs.gaze.tolist(),
                    "emotion": obs.emotion_probs.tolist(),
                    "posture": obs.posture_probs.tolist(),
                    "behavior": POSTURE_LABELS[int(np.argmax(obs.posture_probs))],
                    "has_phone_obj": bool(obs.has_phone_obj),
                    "eye_aspect": float(obs.eye_aspect),
                    "mouth_aspect": float(obs.mouth_aspect),
                    "motion": float(obs.motion),
                }
            )
        return {
            "timestamp": timestamp,
            "students": per_student,
            "events": events,
        }

    def run_video(self, source: str | int, max_seconds: float | None = None):
        fps = float(self.cfg.get("fps", 10))
        for ts, frame in iter_frames(source, target_fps=fps):
            if max_seconds is not None and ts > max_seconds:
                break
            yield self.process_frame(ts, frame)


def _mean_score(tracks: dict[int, TrackState]) -> float | None:
    vals = [t.scores[-1] for t in tracks.values() if t.scores and t.scores[-1] is not None]
    if not vals:
        return None
    return float(np.mean(vals))
