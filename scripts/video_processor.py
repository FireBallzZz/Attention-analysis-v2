"""Video analysis job runner for the upload dashboard.

Runs the full six-stage attention pipeline over an uploaded video and writes:
- an annotated output video (H.264, playable directly in a browser)
- a per-frame scores JSONL
- a summary of attention metrics, drop events and timings

The pipeline is loaded once per worker and reused across jobs, because model
loading costs ~3 seconds and the weights are identical for every upload.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "configs" / "pipeline.yaml"


def _ffmpeg_exe() -> str | None:
    """Locate an ffmpeg binary, preferring the one bundled with imageio-ffmpeg."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    from shutil import which

    return which("ffmpeg")


@dataclass
class JobResult:
    job_id: str
    status: str = "queued"
    stage: str = "Queued"
    progress: float = 0.0
    message: str = ""
    error: str | None = None

    input_name: str = ""
    input_path: str = ""
    output_video: str = ""
    output_jsonl: str = ""
    output_exists: bool = False

    # Timing (seconds)
    queued_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    model_load_s: float = 0.0
    inference_s: float = 0.0
    encode_s: float = 0.0
    total_s: float = 0.0

    # Video facts
    video_duration_s: float = 0.0
    video_fps: float = 0.0
    video_width: int = 0
    video_height: int = 0
    frames_processed: int = 0
    processing_fps: float = 0.0
    realtime_factor: float = 0.0

    # Results
    students_tracked: int = 0
    observations: int = 0
    attentive_pct: float = 0.0
    mean_score: float = 0.0
    drop_events: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    students: list[dict[str, Any]] = field(default_factory=list)
    behaviors: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 4),
            "message": self.message,
            "error": self.error,
            "input_name": self.input_name,
            "output_video": self.output_video,
            "output_jsonl": self.output_jsonl,
            "output_exists": self.output_exists,
            "timing": {
                "model_load_s": round(self.model_load_s, 2),
                "inference_s": round(self.inference_s, 2),
                "encode_s": round(self.encode_s, 2),
                "total_s": round(self.total_s, 2),
                "queued_s": round(self.started_at - self.queued_at, 2) if self.started_at else 0.0,
            },
            "video": {
                "duration_s": round(self.video_duration_s, 2),
                "fps": round(self.video_fps, 2),
                "width": self.video_width,
                "height": self.video_height,
                "frames_processed": self.frames_processed,
                "processing_fps": round(self.processing_fps, 2),
                "realtime_factor": round(self.realtime_factor, 2),
            },
            "results": {
                "students_tracked": self.students_tracked,
                "observations": self.observations,
                "attentive_pct": round(self.attentive_pct, 1),
                "mean_score": round(self.mean_score, 1),
                "drop_events": self.drop_events,
                "timeline": self.timeline,
                "students": self.students,
                "behaviors": self.behaviors,
            },
        }


class VideoProcessor:
    """Serialises jobs and reuses one loaded pipeline across them.

    A single GPU cannot run several attention pipelines concurrently without
    thrashing, so jobs run one at a time in a background thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pipe = None
        self._pipe_error: str | None = None

    def _get_pipeline(self):
        """Load the pipeline once and reuse it for every subsequent job."""
        if self._pipe is not None:
            return self._pipe
        if self._pipe_error:
            raise RuntimeError(self._pipe_error)

        import os
        import sys

        for path in (str(WORKSPACE_ROOT), str(WORKSPACE_ROOT / "src")):
            if path not in sys.path:
                sys.path.insert(0, path)

        # Ultralytics resolves bare weight filenames (e.g. "yolo11n-pose.pt")
        # against the process working directory, so point it at the project
        # root where those files live. Without this, YOLO tries to download
        # them. The CWD is restored immediately after loading.
        previous_cwd = Path.cwd()
        try:
            os.chdir(WORKSPACE_ROOT)
            from attention_pipeline.pipeline import AttentionPipeline
            from attention_pipeline.utils import load_yaml

            cfg = load_yaml(str(CONFIG_PATH))

            # Resolve detector weights to absolute paths so they are found
            # regardless of the working directory.
            detect_cfg = cfg.setdefault("detect", {})
            for key in ("weights", "person_weights"):
                value = detect_cfg.get(key)
                if value:
                    candidate = (WORKSPACE_ROOT / value).resolve()
                    if candidate.exists():
                        detect_cfg[key] = str(candidate)
                    else:
                        raise FileNotFoundError(
                            f"Detector weights not found: {candidate}. "
                            f"Expected '{value}' in the project root."
                        )

            pipe = AttentionPipeline(cfg, load_weights=True)
        finally:
            os.chdir(previous_cwd)

        self._pipe = pipe
        return pipe

    def run(
        self,
        job: JobResult,
        input_path: Path,
        output_video: Path,
        output_jsonl: Path,
        max_seconds: float | None = None,
    ) -> JobResult:
        """Process one video end to end, updating ``job`` as it goes."""
        try:
            job.status = "running"
            job.stage = "Loading models"
            job.progress = 0.02
            job.started_at = time.time()
            t0 = time.time()
            pipe = self._get_pipeline()
            job.model_load_s = time.time() - t0
            job.stage = "Reading video"
            job.progress = 0.08

            # Discard any temporal state left by the previous video. The
            # weights stay loaded, but tracks, track ids and the drop
            # detector's EMA must not carry over.
            pipe.reset()

            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {input_path.name}")

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames <= 0:
                total_frames = 0

            job.video_fps = fps
            job.video_width = width
            job.video_height = height
            job.video_duration_s = total_frames / fps if fps else 0.0

            max_frames = int(max_seconds * fps) if max_seconds else total_frames
            if max_frames <= 0:
                max_frames = total_frames

            output_video.parent.mkdir(parents=True, exist_ok=True)
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)

            # Frame sink: OpenCV writes raw frames, ffmpeg encodes them to
            # H.264 so the result plays in any browser.
            frame_sink = _FrameSink(output_video, fps, (width, height))

            infer_fps = float(pipe.cfg.get("fps", 10))
            infer_dt = 1.0 / max(infer_fps, 1.0)
            next_infer = 0.0

            high = float(pipe.cfg.get("viz", {}).get("high", 70.0))
            low = float(pipe.cfg.get("viz", {}).get("moderate_low", 40.0))

            from attention_pipeline.stages.visualize import attention_band, draw_attention

            last_result: dict[str, Any] = {"timestamp": 0.0, "students": [], "events": []}
            frame_count = 0
            observations = 0
            attentive = 0
            scored_total = 0
            all_scores: list[float] = []
            timeline: list[dict[str, Any]] = []
            behaviors: dict[str, int] = {}
            per_student: dict[int, dict[str, Any]] = {}
            drop_events: list[dict[str, Any]] = []
            seen_events: set[tuple[int, float]] = set()

            jsonl_handle = output_jsonl.open("w", encoding="utf-8")
            t_infer = time.time()

            while frame_count < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_count / fps
                # Fixed 10 fps schedule (same as stages/source.iter_frames). The old
                # `next_infer = timestamp + infer_dt` drifted to 8 fps on 24 fps video
                # because of floating-point rounding, so motion was measured over
                # 0.125 s instead of the 0.1 s the model was evaluated on.
                if timestamp + 1e-6 >= next_infer:
                    last_result = pipe.process_frame(timestamp, frame)
                    next_infer += infer_dt
                    if next_infer <= timestamp:
                        next_infer = timestamp + infer_dt

                    frame_scores: list[float] = []
                    frame_attentive = 0
                    for s in last_result["students"]:
                        observations += 1
                        sid = int(s["student_id"])
                        score = s.get("score")
                        cue = str(s.get("cue") or s.get("behavior") or "")

                        band = s.get("band")
                        if band is None and score is not None:
                            band = attention_band(float(score), high, low)

                        if score is not None:
                            sc = float(score)
                            frame_scores.append(sc)
                            all_scores.append(sc)
                            scored_total += 1
                            if sc >= pipe.binary_threshold * 100.0:
                                frame_attentive += 1
                                attentive += 1

                        behaviour = str(s.get("behavior") or "")
                        if behaviour:
                            behaviors[behaviour] = behaviors.get(behaviour, 0) + 1

                        entry = per_student.setdefault(
                            sid,
                            {
                                "id": sid,
                                "scores": [],
                                "bbox": s.get("bbox"),
                                "behavior": behaviour,
                                "cue": cue,
                            },
                        )
                        if score is not None:
                            entry["scores"].append(float(score))
                        if s.get("bbox"):
                            entry["bbox"] = s["bbox"]

                        entry["cue"] = cue or entry["cue"]
                        entry["behavior"] = behaviour or entry["behavior"]

                    if frame_scores:
                        timeline.append(
                            {
                                "t": round(timestamp, 2),
                                "mean_score": round(float(np.mean(frame_scores)), 1),
                                "attentive": frame_attentive,
                                "distracted": len(frame_scores) - frame_attentive,
                            }
                        )

                    for ev in last_result["events"]:
                        event = ev.get("event")
                        rec = ev.get("recommendation") or {}
                        if event is None:
                            continue
                        key = (int(event.student_id), round(float(event.timestamp), 1))
                        if key in seen_events:
                            continue
                        seen_events.add(key)
                        drop_events.append(
                            {
                                "student_id": int(event.student_id),
                                "timestamp": round(float(event.timestamp), 2),
                                "score": round(float(event.score), 1),
                                "kind": event.kind,
                                "modality": event.dominant_modality,
                                "explanation": rec.get("explanation", ""),
                                "recommendation": rec.get("teaching_recommendation", ""),
                                "suggestion": rec.get("intervention_suggestion", ""),
                            }
                        )

                    jsonl_handle.write(
                        json.dumps(
                            {
                                "frame": frame_count,
                                "timestamp": round(timestamp, 3),
                                "students": [
                                    {
                                        **s,
                                        "band": s.get("band")
                                        or (
                                            attention_band(float(s["score"]), high, low)
                                            if s.get("score") is not None
                                            else "WARMUP"
                                        ),
                                    }
                                    for s in last_result["students"]
                                ],
                                "drops": len(last_result["events"]),
                            }
                        )
                        + "\n"
                    )

                drawn = draw_attention(frame, last_result, high=high, low=low)
                frame_sink.write(drawn)
                frame_count += 1

                if total_frames:
                    job.progress = 0.08 + 0.82 * (frame_count / total_frames)
                job.stage = f"Analysing frame {frame_count:,} / {max_frames:,}"
                job.message = f"{frame_count/max_frames*100:.0f}% of video processed"

                if frame_count % 30 == 0:
                    job.processing_fps = frame_count / max(time.time() - t_infer, 1e-6)

            cap.release()
            jsonl_handle.close()
            job.inference_s = time.time() - t_infer
            job.frames_processed = frame_count
            job.processing_fps = frame_count / max(job.inference_s, 1e-6)
            video_secs = frame_count / fps if fps else 0.0
            job.realtime_factor = video_secs / job.inference_s if job.inference_s else 0.0

            job.stage = "Encoding output video"
            job.progress = 0.93
            t_enc = time.time()
            frame_sink.close()
            job.encode_s = time.time() - t_enc

            # Build per-student summary.
            students_out = []
            for sid in sorted(per_student):
                entry = per_student[sid]
                scores = entry["scores"]
                if not scores:
                    continue
                arr = np.array(scores)
                students_out.append(
                    {
                        "id": sid,
                        "mean_score": round(float(arr.mean()), 1),
                        "min_score": round(float(arr.min()), 1),
                        "max_score": round(float(arr.max()), 1),
                        "attentive_pct": round(float((arr >= 50).sum()) / len(arr) * 100, 1),
                        "behavior": entry["behavior"],
                        "cue": entry["cue"],
                        "bbox": entry["bbox"],
                    }
                )

            job.students = students_out
            job.students_tracked = len(students_out)
            job.observations = observations
            job.mean_score = round(float(np.mean(all_scores)), 1) if all_scores else 0.0
            job.attentive_pct = (
                round(attentive / scored_total * 100, 1) if scored_total else 0.0
            )
            job.timeline = timeline
            job.behaviors = dict(sorted(behaviors.items(), key=lambda kv: -kv[1]))
            job.drop_events = sorted(drop_events, key=lambda e: e["timestamp"])

            job.output_video = output_video.name
            job.output_jsonl = output_jsonl.name
            job.output_exists = output_video.exists() and output_video.stat().st_size > 0
            job.status = "done"
            job.stage = "Complete"
            job.progress = 1.0
            job.message = (
                f"Analysed {video_secs:.1f}s of video in {job.inference_s:.1f}s "
                f"({job.realtime_factor:.2f}x realtime)"
            )

        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            job.status = "error"
            job.stage = "Failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.message = job.error
        finally:
            job.finished_at = time.time()
            job.total_s = job.finished_at - (job.started_at or job.finished_at)

        return job


class _FrameSink:
    """Buffers drawn frames and encodes them to H.264 with ffmpeg.

    OpenCV's built-in ``mp4v`` writer produces MPEG-4 Part 2, which Chrome and
    Firefox refuse to play. Piping raw frames through ffmpeg's libx264
    produces a file every browser can play inline.
    """

    def __init__(self, path: Path, fps: float, size: tuple[int, int]) -> None:
        self.path = path
        self.fps = fps
        self.size = size
        self._proc: subprocess.Popen | None = None
        self._fallback: cv2.VideoWriter | None = None
        self._exe = _ffmpeg_exe()

        if self._exe:
            cmd = [
                self._exe,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{size[0]}x{size[1]}",
                "-r", f"{fps:.6f}",
                "-i", "-",
                "-an",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(path),
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                self._proc = None

        if self._proc is None:
            # No ffmpeg available: fall back to OpenCV's writer so the job
            # still produces a downloadable file.
            self._fallback = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
            )

    def write(self, frame: np.ndarray) -> None:
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
                return
            except (BrokenPipeError, OSError):
                self._proc = None
        if self._fallback is not None:
            self._fallback.write(frame)

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=60)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._fallback is not None:
            self._fallback.release()
            self._fallback = None
