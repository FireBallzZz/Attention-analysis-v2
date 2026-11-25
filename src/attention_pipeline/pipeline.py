"""End-to-end runner for the six-stage attention pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from attention_pipeline.features import to_student_frame
from attention_pipeline.rubric import (
    SpeechPersistenceTracker,
    apply_peer_cues,
    classify_attention,
)
from attention_pipeline.models.attention_fusion import AttentionFusionNet
from attention_pipeline.models.drop_detector import AttentionDropDetector, DropConfig
from attention_pipeline.models.l2cs_net import L2CSNet
from attention_pipeline.models.manual_binary import ManualBinaryAttentionClassifier
from attention_pipeline.models.posture_mlp import PostureMLP
from attention_pipeline.stages.attention import AttentionScorer
from attention_pipeline.stages.detect_track import DetectorTracker
from attention_pipeline.stages.intervention import recommend
from attention_pipeline.stages.modalities import FaceModalityExtractor, PostureExtractor
from attention_pipeline.stages.pitch_baseline import EyeBaselineTracker, PitchBaselineTracker
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
        # Optional motion-based attention model (the cue that transfers between
        # classrooms; see scripts/motion_rule.py). When configured it replaces
        # the neural score; the hard-evidence rubric rules still apply on top.
        self.motion_scorer = None
        self._prev_frame: tuple[float, np.ndarray] | None = None
        # DAiSEE "individual" model: the network trained only on DAiSEE (one
        # learner per video), applied to each student on their own. Reported per
        # student (engagement, boredom, confusion, frustration). Not used in the
        # attention score: leave-one-classroom-out it is at chance on classroom
        # students and lowers the score's accuracy (reports/improvement/ensemble.json).
        self.daisee_net = None
        dz = project_path(models_cfg.get("fusion_daisee_base")) if load_weights else None
        if dz is not None and dz.exists():
            try:
                self.daisee_net = AttentionFusionNet(
                    feat_dim=fusion_cfg.get("feat_dim", 32), hidden=fusion_cfg.get("hidden", 64)
                ).to(self.device)
                self.daisee_net.load(dz, map_location=self.device)
            except Exception as exc:
                print(f"[pipeline] DAiSEE model not loaded: {exc}")
                self.daisee_net = None
        mr = project_path(models_cfg.get("motion_rule")) if load_weights else None
        if mr is not None and mr.exists():
            from attention_pipeline.motion import LiveMotionScorer, MotionRule

            self.motion_scorer = LiveMotionScorer(MotionRule.load(mr))
            self.scorer.use_heuristic = False
        self.binary_threshold = float(fusion_cfg.get("binary_threshold", 0.50))
        # How the model score and the heuristic rubric are combined.
        # "model_hard" is the default: the rubric may only floor the model
        # score for verified hard evidence of disengagement.
        self.combine_mode = str(fusion_cfg.get("combine_mode", "model_hard"))
        self.heuristic_weight = float(fusion_cfg.get("heuristic_weight", 0.6))
        # SCB confidence (hand-raising / reading / writing) above which a student's
        # score is not allowed below 50 in model_floor mode; 0 disables it.
        self.study_protect = float(fusion_cfg.get("study_protect", 0.0))
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
        self._prev_resized_crops: dict[int, tuple[float, np.ndarray]] = {}
        self._phone_boxes_recent: list[tuple[int, float, float, float]] = []
        self._phone_hits_recent: dict[int, list[tuple[float, float]]] = {}
        self._phone_hold_until: dict[int, float] = {}
        self._talk_streak: dict[int, int] = {}
        self._frame_count: int = 0
        self._lesson_yaw: float | None = None
        self._lesson_gaze_yaw: float | None = None
        self.pitch_baseline = PitchBaselineTracker()
        self.eye_baseline = EyeBaselineTracker()
        self.speech_tracker = SpeechPersistenceTracker(required=4, decay=1)
        self.dataset_status = {
            "l2cs_gaze": loaded_gaze,
            "scb_behavior": self.scb.enabled,
            "scb_yaml": bool(self.scb_data_yaml and self.scb_data_yaml.exists()),
            "fusion": loaded_fusion,
        }

    @classmethod
    def from_yaml(cls, path: str | Path, load_weights: bool = True) -> "AttentionPipeline":
        return cls(load_yaml(str(path)), load_weights=load_weights)
    def reset(self) -> None:
        """Clear all per-video state so the pipeline can score a new video.

        The model weights stay loaded — only the temporal state is discarded.
        Without this, tracks, track ids and the drop detector's EMA carry over
        from the previous video and contaminate the next one.
        """
        self.tracks.clear()
        self._prev_crops.clear()
        self._prev_resized_crops.clear()
        self._phone_boxes_recent.clear()
        self._phone_hits_recent.clear()
        self._phone_hold_until.clear()
        self._talk_streak.clear()
        self._frame_count = 0
        self._prev_frame = None
        if self.motion_scorer is not None:
            self.motion_scorer.reset()
        self._lesson_yaw = None
        self._lesson_gaze_yaw = None
        self.pitch_baseline.reset()
        self.eye_baseline.reset()
        self.speech_tracker.reset()
        self.drop.reset()
        self.detector.reset()
        self.faces.reset()
        self.scb.reset()


    def _maybe_load(self, module, path: str | None) -> bool:
        p = project_path(path)
        if p is None or not p.exists():
            return False
        try:
            module.load(p, map_location=self.device)
        except Exception as exc:
            # A checkpoint that does not match this module's architecture must
            # not abort the whole pipeline: the caller falls back to the next
            # candidate, or to the heuristic rubric when none is left.
            print(f"[pipeline] could not load {p.name} into {type(module).__name__}: {exc}")
            return False
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

        # Current-frame phone detections only (avoid multiplying a single fluke via stale boxes)
        self._phone_boxes_recent.clear()
        if getattr(self, "phone_model", None) is not None:
            try:
                p_res = self.phone_model.predict(
                    frame, conf=0.40, imgsz=1280, verbose=False, device=self.device
                )[0]
                for bx, conf in _verified_phone_boxes(p_res, frame_h=frame.shape[0]):
                    self._phone_boxes_recent.append(
                        (
                            self._frame_count,
                            float((bx[0] + bx[2]) / 2.0),
                            float((bx[1] + bx[3]) / 2.0),
                            float(conf),
                        )
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
                self._prev_resized_crops.pop(tid, None)
            if len(self._prev_crops) > 100:
                oldest_tids = sorted(self._prev_crops.keys(), key=lambda k: self._prev_crops[k][0])
                for tid in oldest_tids[: len(self._prev_crops) - 50]:
                    del self._prev_crops[tid]
                    self._prev_resized_crops.pop(tid, None)

        observed: list[tuple[TrackState, Any]] = []
        phone_meta: list[tuple[TrackState, Any, float, float, np.ndarray]] = []
        pending_phone_crops: list[tuple[np.ndarray, int, Any, int, int]] = []
        phone_raw_confs: dict[int, float] = {}
        all_det_boxes = [d.bbox_xyxy for d in detections]
        for idx_det, (det, scb_probs) in enumerate(zip(detections, scb_matches)):
            track = self.tracks.setdefault(det.track_id, TrackState(student_id=det.track_id))
            track.last_bbox = det.bbox_xyxy
            suppress_mouth = _mouth_occluded(det, all_det_boxes, idx_det)
            face = self.faces.extract(
                frame, det, det.track_id, suppress_mouth=suppress_mouth
            )

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

            # Jitter-resilient body motion for rubric (disturbing / stillness) without changing feat[18]
            body_motion: float | None = None
            if crop.size > 0:
                resized_crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
                if det.track_id in self._prev_resized_crops:
                    prev_r_ts, prev_rc = self._prev_resized_crops[det.track_id]
                    if 0.0 <= (timestamp - prev_r_ts) <= 0.5:
                        body_motion = float(np.mean(cv2.absdiff(resized_crop, prev_rc)) / 255.0)
                self._prev_resized_crops[det.track_id] = (timestamp, resized_crop)

            posture = self.posture_x.predict(det.keypoints, det.bbox_xyxy)
            if scb_probs is not None:
                # Blend SCB with geometric pose so one wrong SCB box cannot dominate.
                posture = (blend * scb_probs + (1.0 - blend) * posture).astype(np.float32)
                posture = posture / max(float(posture.sum()), 1e-6)

            # Per-student head-pitch baseline. Absolute pitch carries a large
            # systematic offset, so the rubric needs each student's own
            # neutral to tell "looking down at notes" from "head on desk".
            head_pitch = float(face.head_pose[1])
            if face.pose_valid:
                self.pitch_baseline.update(det.track_id, head_pitch)
                pitch_delta = self.pitch_baseline.delta(det.track_id, head_pitch)
            else:
                pitch_delta = None

            # Match full-frame phone box to student below-shoulder lap/hand zone (exclude idle objects far below sy2)
            sx1, sy1, sx2, sy2 = det.bbox_xyxy
            sbw = sx2 - sx1
            sbh = sy2 - sy1
            sh_floor_y = _phone_min_y(det.keypoints, sy1, sbh)
            wrist_pts = []
            if det.keypoints is not None and det.keypoints.shape[0] > 10:
                for wi in (9, 10):
                    if float(det.keypoints[wi, 2]) >= 0.25:
                        wrist_pts.append(det.keypoints[wi, :2])
            own_radius = 0.28 * max(sbw, sbh)
            raw_phone_conf = 0.0
            for _, px, py, conf in self._phone_boxes_recent:
                if py < sh_floor_y:
                    continue
                in_box = sx1 <= px <= sx2 and py <= (sy2 + 0.05 * sbh)
                near_wr = any(
                    float(np.hypot(px - float(wp[0]), py - float(wp[1]))) <= own_radius
                    for wp in wrist_pts
                ) and py <= (sy2 + 0.18 * sbh)
                if in_box or near_wr:
                    raw_phone_conf = max(raw_phone_conf, conf)
            phone_raw_confs[det.track_id] = raw_phone_conf

            if (
                raw_phone_conf == 0.0
                and getattr(self, "phone_model", None) is not None
                and sbh >= 70.0
                and sbw >= 32.0
                and head_pitch >= 15.0
                and float(posture[2]) < 0.45
            ):
                py1 = max(0, int(sy1 + 0.35 * sbh))
                py2 = min(frame.shape[0], int(sy2 + 0.15 * sbh))
                px1 = max(0, int(sx1 - 0.12 * sbw))
                px2 = min(frame.shape[1], int(sx2 + 0.12 * sbw))
                if py2 > py1 and px2 > px1:
                    desk_crop = frame[py1:py2, px1:px2]
                    if desk_crop.size > 0:
                        pending_phone_crops.append((desk_crop, det.track_id, det, px1, py1))

            # Per-student eye baseline: a closed eye is a drop relative to the
            # student's own normal opening, not an absolute threshold.
            self.eye_baseline.update(det.track_id, face.eye_openness)
            eye_closed = self.eye_baseline.is_closed(det.track_id, face.eye_openness)

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
                has_phone_obj=False,
                mouth_motion=float(face.mouth_motion),
                eye_closed=bool(eye_closed),
                mouth_active=bool(face.mouth_motion >= 0.02),
                aspects_valid=bool(face.aspects_valid and eye_closed is not None),
                pose_reliable=bool(face.pose_valid),
                pitch_delta=pitch_delta,
                eye_openness=face.eye_openness,
                mouth_openness=face.mouth_openness,
                body_motion=body_motion,
            )

            observed.append((track, obs))
            phone_meta.append((track, obs, raw_phone_conf, head_pitch, posture))

        # Batched crop-level phone verification across only downward-looking students
        if pending_phone_crops and getattr(self, "phone_model", None) is not None:
            try:
                c_results = self.phone_model.predict(
                    [item[0] for item in pending_phone_crops],
                    conf=0.30,
                    imgsz=320,
                    verbose=False,
                    device=self.device,
                )
                for (_crop, tid, det, px1, py1), c_res in zip(pending_phone_crops, c_results):
                    sx1, sy1, sx2, sy2 = det.bbox_xyxy
                    sbw, sbh = sx2 - sx1, sy2 - sy1
                    sh_floor_y = _phone_min_y(det.keypoints, sy1, sbh)
                    wrist_pts = []
                    if det.keypoints is not None and det.keypoints.shape[0] > 10:
                        for wi in (9, 10):
                            if float(det.keypoints[wi, 2]) >= 0.25:
                                wrist_pts.append(det.keypoints[wi, :2])
                    own_radius = 0.28 * max(sbw, sbh)
                    best_conf = 0.0
                    for bx, conf in _verified_phone_boxes(
                        c_res, frame_h=frame.shape[0], offset_y=float(py1)
                    ):
                        if conf < 0.50:
                            continue
                        cx_abs = px1 + float((bx[0] + bx[2]) / 2.0)
                        cy_abs = py1 + float((bx[1] + bx[3]) / 2.0)
                        if cy_abs < sh_floor_y:
                            continue
                        in_core = (
                            sx1 <= cx_abs <= sx2
                            and sh_floor_y <= cy_abs <= (sy2 + 0.05 * sbh)
                        )
                        near_wrist = any(
                            float(np.hypot(cx_abs - float(wp[0]), cy_abs - float(wp[1]))) <= own_radius
                            for wp in wrist_pts
                        ) and cy_abs <= (sy2 + 0.18 * sbh)
                        if in_core or near_wrist:
                            best_conf = max(best_conf, conf)
                    if best_conf > 0.0:
                        phone_raw_confs[tid] = max(phone_raw_confs.get(tid, 0.0), best_conf)
            except Exception:
                pass

        # Temporal confirmation (>= 0.70 single hit or >= 3 hits within 0.8s) + 0.5s occlusion bridge
        for track, obs, _init_conf, head_pitch, posture in phone_meta:
            tid = obs.student_id
            raw_conf = phone_raw_confs.get(tid, 0.0)
            hits = [
                (t, c)
                for t, c in self._phone_hits_recent.get(tid, [])
                if 0.0 <= (timestamp - t) <= 0.8
            ]
            if raw_conf > 0.0:
                hits.append((float(timestamp), float(raw_conf)))
            self._phone_hits_recent[tid] = hits

            confirmed = raw_conf >= 0.70 or len(hits) >= 3
            has_phone_obj = False
            if confirmed:
                has_phone_obj = True
                self._phone_hold_until[tid] = float(timestamp) + 0.5
            elif head_pitch <= 15.0 or float(posture[2]) >= 0.45:
                self._phone_hold_until.pop(tid, None)
            elif float(timestamp) <= self._phone_hold_until.get(tid, -1e9):
                has_phone_obj = True
            obs.has_phone_obj = has_phone_obj

        if observed:
            med = float(np.median([obs.head_pose[0] for _, obs in observed]))
            med_g = float(np.median([obs.gaze[0] for _, obs in observed]))
            if self._lesson_yaw is None:
                self._lesson_yaw = med
            else:
                self._lesson_yaw = 0.2 * med + 0.8 * self._lesson_yaw
            if self._lesson_gaze_yaw is None:
                self._lesson_gaze_yaw = med_g
            else:
                self._lesson_gaze_yaw = 0.2 * med_g + 0.8 * self._lesson_gaze_yaw
            for _, obs in observed:
                obs.rel_gaze_yaw = ((float(obs.gaze[0]) - self._lesson_gaze_yaw + 180.0) % 360.0) - 180.0

        per_student = []
        events = []
        pending = []
        for track, obs in observed:
            # Feed the live speech evidence into the persistence tracker so a
            # "talking to peer" verdict needs sustained movement, not one frame.
            self.speech_tracker.update(obs.student_id, bool(obs.mouth_active))
            score, rubric, fusion_score = self.scorer.push(
                track, obs.features, timestamp, obs=obs, lesson_yaw=self._lesson_yaw
            )
            # Per-student component scores, reported in the output (see scripts/collect_components.py)
            comp = {"neural_score": None if fusion_score is None else float(fusion_score),
                    "daisee_score": None, "daisee_aux": None, "motion_score": None}
            if self.daisee_net is not None and len(track.feature_window) >= self.scorer.window:
                with torch.inference_mode():
                    xw = torch.from_numpy(np.stack(track.feature_window[-self.scorer.window:], 0)).unsqueeze(0).to(self.device)
                    ds, aux = self.daisee_net(xw)
                comp["daisee_score"] = float(ds.item())
                comp["daisee_aux"] = [round(float(v), 3) for v in aux[0].tolist()]
            if self.motion_scorer is not None:
                from attention_pipeline.motion import box_motion

                prev = None
                if self._prev_frame is not None and 0.0 < timestamp - self._prev_frame[0] <= 0.5:
                    prev = self._prev_frame[1]
                ms = self.motion_scorer.update(int(obs.student_id), box_motion(prev, frame, obs.bbox))
                comp["motion_score"] = ms
                score = fusion_score = ms
                if track.scores:
                    track.scores[-1] = ms
                if getattr(track, "fusion_scores", None):
                    track.fusion_scores[-1] = ms
            obs.components = comp
            pending.append((track, obs, score, rubric, fusion_score))

        peer = apply_peer_cues(
            [obs for _track, obs, _score, _rubric, _fs in pending],
            [rubric for _track, _obs, _score, rubric, _fs in pending],
            self._lesson_yaw,
            self.speech_tracker,
        )
        debounced_peer = []
        active_talk_ids = set()
        for (_t, obs, _sc, solo_rubric, _fs), peer_rubric in zip(pending, peer):
            tid = int(obs.student_id)
            if peer_rubric.cue == "talking to peer":
                streak = self._talk_streak.get(tid, 0) + 1
                self._talk_streak[tid] = streak
                active_talk_ids.add(tid)
                debounced_peer.append(peer_rubric if streak >= 2 else solo_rubric)
            else:
                self._talk_streak.pop(tid, None)
                debounced_peer.append(peer_rubric)
        for stale_tid in [k for k in self._talk_streak if k not in active_talk_ids]:
            del self._talk_streak[stale_tid]

        for (track, obs, score, _old, fusion_score), rubric in zip(pending, debounced_peer):
            # Multimodal Action-Aware Neural Attention Scoring:
            # If verified active academic behavior (reading/writing/hand-raising) is detected by the rubric,
            # head pitch down corresponds to productive desk study, not disengagement.
            if fusion_score is not None:
                effective_score = self._combine_scores(
                    float(fusion_score), rubric, obs
                )
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
                    "behavior": (
                        "neutral"
                        if float(np.ptp(obs.posture_probs)) < 1e-4
                        else POSTURE_LABELS[int(np.argmax(obs.posture_probs))]
                    ),
                    "has_phone_obj": bool(obs.has_phone_obj),
                    "eye_aspect": float(obs.eye_aspect),
                    "mouth_aspect": float(obs.mouth_aspect),
                    "motion": float(obs.motion),
                    **getattr(obs, "components", {}),
                }
            )
        if self.motion_scorer is not None:
            self._prev_frame = (timestamp, frame.copy())
        return {
            "timestamp": timestamp,
            "students": per_student,
            "events": events,
        }

    # Cues that are verified hard evidence of disengagement. The rubric may
    # only lower the model score for these, never raise it.
    _HARD_EVIDENCE_CUES = (
        "using phone",
        "phone-like pose",
        "sleeping",
        "head on table",
        "talking to peer",
        "disturbing others",
        "looking left",
        "looking right",
        "looking away",
    )

    # Active academic learning cues. These represent productive desk study
    # (reading, writing, taking notes, raising hand) and must not be penalized
    # by webcam pretraining bias (head pitch down).
    _ACTIVE_STUDY_CUES = (
        "reading",
        "writing",
        "taking notes",
        "notes",
        "hand raised",
        "standing / presenting",
    )

    def _combine_scores(self, fusion_score: float, rubric, obs) -> float:
        """Combine the trained model's score with the heuristic rubric.

        Measured against the human annotation on all three classroom clips,
        the rubric's own score is INVERTED (point-biserial r = -0.35) while
        the model's score is well ordered (r = +0.61). Blending the two
        freely therefore washes the model's signal out.

        Under `model_hard`:
        - Hard disengagement (phone, sleep, head on desk, peer talk) FLOORS
          the score to prevent false engagement.
        - Active academic behaviors (reading, writing, taking notes, hand raised)
          LIFT the score so productive desk study is never penalized by
          webcam pretraining domain gap (downward head pitch / gaze).
        - Soft / neutral states use the model score directly.
        """
        mode = self.combine_mode
        h_score = float(rubric.score)
        cue_str = str(rubric.cue).lower()

        if mode == "model":
            return float(fusion_score)

        if mode == "model_floor":
            # Hard disengagement evidence may only LOWER the score. The rubric's
            # head-pose "active study" lift is not used (it was reversed on human
            # labels). Instead, when the SCB detector is CONFIDENT a student is
            # reading, writing or raising a hand, motion from that activity must
            # not push them below the HIGH/LOW line (study protection; raises
            # unseen-room balanced accuracy 63.1% -> 64.9%, harms no room).
            if any(word in cue_str for word in self._HARD_EVIDENCE_CUES):
                return min(float(fusion_score), h_score)
            if self.study_protect > 0 and obs is not None and getattr(obs, "posture_probs", None) is not None:
                pp = np.asarray(obs.posture_probs, dtype=float).reshape(-1)
                if pp.size >= 3 and float(pp[:3].max()) >= self.study_protect:
                    return max(float(fusion_score), 50.0)
            return float(fusion_score)

        if mode == "model_hard":
            if any(word in cue_str for word in self._HARD_EVIDENCE_CUES):
                return min(float(fusion_score), h_score)
            if any(word in cue_str for word in self._ACTIVE_STUDY_CUES):
                return max(float(fusion_score), h_score)
            return float(fusion_score)

        # "blended": the previous behaviour, kept for comparison.
        is_active_study = (
            h_score >= 70.0
            and any(w in cue_str for w in ("reading", "writing", "hand", "watching",
                                           "presenting", "oriented"))
            and not any(w in cue_str for w in ("phone", "sleep", "table", "left",
                                               "right", "talk", "disturb", "away"))
        )
        is_disengaged = (
            rubric.band == "LOW"
            or h_score <= 40.0
 or any(w in cue_str for w in ("phone", "sleep", "table", "talk",
                                         "disturb", "left", "right", "away"))
        )
        weight = float(self.heuristic_weight)
        if is_disengaged:
            return min(float(fusion_score), h_score)
        if is_active_study:
            return float((1.0 - weight) * fusion_score + weight * h_score)
        return float(fusion_score)

    def run_video(self, source: str | int, max_seconds: float | None = None):
        fps = float(self.cfg.get("fps", 10))
        for ts, frame in iter_frames(source, target_fps=fps):
            if max_seconds is not None and ts > max_seconds:
                break
            yield self.process_frame(ts, frame)


_NON_PHONE_COCO_IDS = {39, 41, 63, 64, 65, 73}  # bottle, cup, laptop, mouse, remote, book


def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = aa + bb - inter
    return inter / den if den > 0.0 else 0.0


def _verified_phone_boxes(
    yolo_res: Any,
    frame_h: int | None = None,
    offset_y: float = 0.0,
) -> list[tuple[np.ndarray, float]]:
    """Return cell-phone detections (COCO 67) not outperformed by overlapping bottle/cup/book."""
    boxes = getattr(yolo_res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    phones: list[tuple[np.ndarray, float]] = []
    non_phones: list[tuple[np.ndarray, float]] = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy()
        if cls_id == 67:
            phones.append((xyxy, conf))
        elif cls_id in _NON_PHONE_COCO_IDS:
            non_phones.append((xyxy, conf))
    kept: list[tuple[np.ndarray, float]] = []
    for p_box, p_conf in phones:
        bw = max(float(p_box[2] - p_box[0]), 0.0)
        bh = max(float(p_box[3] - p_box[1]), 0.0)
        hi, lo = max(bw, bh), min(bw, bh)
        # Reject tiny pencil/pen/edge slivers and bottom-edge paper strips clipped at frame border
        if hi < 25.5 or lo < 9.5 or (hi / max(lo, 1e-6)) > 4.2:
            continue
        if frame_h is not None and (float(p_box[3]) + offset_y) >= float(frame_h) - 3.0 and bh < 20.0:
            continue
        if any(
            _box_iou_xyxy(p_box, o_box) >= 0.25 and o_conf >= (p_conf - 0.05)
            for o_box, o_conf in non_phones
        ):
            continue
        kept.append((p_box, p_conf))
    return kept


def _mouth_occluded(det: Any, all_boxes: list[np.ndarray], self_idx: int) -> bool:
    """True when a foreground/walking person overlaps this student's mouth region."""
    sx1, sy1, sx2, sy2 = [float(v) for v in det.bbox_xyxy]
    sbw = max(sx2 - sx1, 1.0)
    sbh = max(sy2 - sy1, 1.0)
    kpts = getattr(det, "keypoints", None)
    if (
        kpts is not None
        and kpts.shape[0] >= 3
        and float(kpts[0, 2]) >= 0.2
        and (float(kpts[1, 2]) >= 0.2 or float(kpts[2, 2]) >= 0.2)
    ):
        mx = float(kpts[0, 0])
        my = float(kpts[0, 1]) + 0.06 * sbh
        m_box = (mx - 0.16 * sbw, my - 0.10 * sbh, mx + 0.16 * sbw, my + 0.12 * sbh)
    else:
        m_box = (sx1 + 0.18 * sbw, sy1 + 0.14 * sbh, sx2 - 0.18 * sbw, sy1 + 0.42 * sbh)
    m_area = max((m_box[2] - m_box[0]) * (m_box[3] - m_box[1]), 1.0)
    self_area = sbw * sbh
    for j, ob in enumerate(all_boxes):
        if j == self_idx:
            continue
        ox1, oy1, ox2, oy2 = [float(v) for v in ob]
        inter = max(0.0, min(m_box[2], ox2) - max(m_box[0], ox1)) * max(
            0.0, min(m_box[3], oy2) - max(m_box[1], oy1)
        )
        if inter / m_area < 0.25:
            continue
        o_area = max(ox2 - ox1, 0.0) * max(oy2 - oy1, 0.0)
        if o_area >= 0.85 * self_area or oy2 >= sy2 - 0.10 * sbh:
            return True
    return False


def _phone_min_y(keypoints: np.ndarray | None, sy1: float, sbh: float) -> float:
    """Minimum image-y for a desk/chest phone so pens near lips/chin are excluded."""
    floor_y = sy1 + 0.52 * sbh
    if keypoints is not None and keypoints.shape[0] > 6:
        if float(keypoints[5, 2]) >= 0.2 and float(keypoints[6, 2]) >= 0.2:
            sh_y = float(keypoints[5, 1] + keypoints[6, 1]) / 2.0
            floor_y = max(floor_y, sh_y - 0.05 * sbh)
    return floor_y


def _mean_score(tracks: dict[int, TrackState]) -> float | None:
    vals = [t.scores[-1] for t in tracks.values() if t.scores and t.scores[-1] is not None]
    if not vals:
        return None
    return float(np.mean(vals))
