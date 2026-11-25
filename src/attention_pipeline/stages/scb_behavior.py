"""SCB-dataset student behavior detector (YOLO boxes + class names)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attention_pipeline.types import SCB_BEHAVIOR_LABELS

# Official SCB-dataset3 names. The HF 3-class zip uses the first three.
SCB3_NAMES = list(SCB_BEHAVIOR_LABELS)
SCB3_TO_HIGH = {"hand-raising", "reading", "writing"}
SCB3_TO_LOW = {"using-phone", "leaning-over-table"}
ROOT = Path(__file__).resolve().parents[3]
YOLOV7_DIR = ROOT / "third_party" / "yolov7"


@dataclass
class SCBDet:
    bbox_xyxy: np.ndarray
    cls: int
    name: str
    conf: float


class SCBBehaviorDetector:
    """SCB-dataset detector. Official HF weights are YOLOv7; YOLO11 works after retrain."""

    def __init__(
        self,
        weights: str | Path | None,
        device: str = "cpu",
        conf: float = 0.25,
        names: list[str] | None = None,
    ) -> None:
        self.weights = Path(weights) if weights else None
        self.device = device
        self.conf = conf
        self.names = names or SCB3_NAMES
        self._model = None
        self._backend = "ultralytics"
        self._stride = 32
        # Maps this checkpoint's own class index -> posture channel index.
        # Resolved by class NAME, never by the checkpoint's class index.
        self._channel_of: dict[int, int] = {}
        self.enabled = bool(self.weights and self.weights.exists())

    def reset(self) -> None:
        """No per-video state to clear (the model stays loaded)."""

    def _load(self) -> None:
        if self._model is not None or not self.enabled:
            return
        path = str(self.weights)
        if _is_ultralytics_ckpt(path):
            from ultralytics import YOLO

            self._model = YOLO(path)
            self._backend = "ultralytics"
            self.names = _names_from_model(self._model)
            self._rebuild_channel_map()
            return
        self._model = _load_yolov7(path, self.device)
        self._backend = "yolov7"
        self._stride = int(self._model.stride.max())
        raw = self._model.module.names if hasattr(self._model, "module") else self._model.names
        if isinstance(raw, dict):
            self.names = [str(raw[i]) for i in range(len(raw))]
        elif raw:
            self.names = [str(n) for n in raw]
        self._rebuild_channel_map()

    def _rebuild_channel_map(self) -> None:
        """Map this checkpoint's class names onto the 6-slot posture vector.

        The deployed checkpoint carries three classes and the merged
        checkpoint carries four, and a checkpoint may name its classes
        differently from the pipeline taxonomy. The posture channel is
        therefore resolved by NAME; a class the checkpoint does not know
        simply has no channel and stays zero.
        """
        self._channel_of = {}
        for cls_id, raw_name in enumerate(self.names):
            canon = _canon_name(str(raw_name))
            if canon in SCB_BEHAVIOR_LABELS:
                self._channel_of[cls_id] = SCB_BEHAVIOR_LABELS.index(canon)

    def infer(self, frame: np.ndarray) -> list[SCBDet]:
        if not self.enabled:
            return []
        self._load()
        if self._backend == "yolov7":
            return self._infer_yolov7(frame)
        out = self._model.predict(
            frame, conf=self.conf, verbose=False, device=self.device
        )[0]
        if out.boxes is None or len(out.boxes) == 0:
            return []
        names = out.names or {i: n for i, n in enumerate(self.names)}
        dets = []
        xyxy = out.boxes.xyxy.cpu().numpy()
        cls = out.boxes.cls.cpu().numpy().astype(int)
        confs = out.boxes.conf.cpu().numpy()
        for box, c, p in zip(xyxy, cls, confs):
            raw = names.get(int(c), self.names[int(c)] if int(c) < len(self.names) else str(c))
            dets.append(
                SCBDet(
                    bbox_xyxy=box.astype(np.float32),
                    cls=int(c),
                    name=_canon_name(str(raw)),
                    conf=float(p),
                )
            )
        return dets

    def _infer_yolov7(self, frame: np.ndarray) -> list[SCBDet]:
        import torch

        _ensure_yolov7_path()
        from utils.datasets import letterbox
        from utils.general import non_max_suppression, scale_coords

        img = letterbox(frame, 640, stride=self._stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).float() / 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            pred = self._model(tensor)[0]
        pred = non_max_suppression(pred, self.conf, 0.45)
        dets: list[SCBDet] = []
        for det in pred:
            if det is None or len(det) == 0:
                continue
            det = det.clone()
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det.cpu().numpy():
                c = int(cls)
                raw = self.names[c] if c < len(self.names) else str(c)
                dets.append(
                    SCBDet(
                        bbox_xyxy=np.array(xyxy, np.float32),
                        cls=c,
                        name=_canon_name(str(raw)),
                        conf=float(conf),
                    )
                )
        return dets

    def match_probs(
        self,
        student_xyxy: np.ndarray,
        dets: list[SCBDet],
        *,
        min_iou: float = 0.4,
        used: set[int] | None = None,
    ) -> np.ndarray | None:
        if not dets:
            return None
        best_i, best_iou = -1, 0.0
        for i, d in enumerate(dets):
            if used is not None and i in used:
                continue
            iou = _iou(student_xyxy, d.bbox_xyxy)
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i < 0 or best_iou < min_iou:
            return None
        if used is not None:
            used.add(best_i)
        return _probs_from_det(dets[best_i], self._channel_of)

    def match_all(
        self,
        student_boxes: list[np.ndarray],
        dets: list[SCBDet],
        *,
        min_iou: float = 0.4,
    ) -> list[np.ndarray | None]:
        """One-to-one SCB assignment so one reading box cannot label every student."""
        if not dets or not student_boxes:
            return [None] * len(student_boxes)
        pairs: list[tuple[float, int, int]] = []
        for si, box in enumerate(student_boxes):
            s_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            scx = 0.5 * float(box[0] + box[2])
            scy = 0.5 * float(box[1] + box[3])
            for di, d in enumerate(dets):
                db = d.bbox_xyxy
                iou = _iou(box, db)
                if iou >= min_iou:
                    pairs.append((iou, si, di))
                elif s_area > 0 and db[0] <= scx <= db[2] and db[1] <= scy <= db[3]:
                    # SCB labels full seated student+desk while pose boxes wrap upper torso.
                    ix1 = max(float(box[0]), float(db[0]))
                    iy1 = max(float(box[1]), float(db[1]))
                    ix2 = min(float(box[2]), float(db[2]))
                    iy2 = min(float(box[3]), float(db[3]))
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    if (inter / s_area) >= 0.75 and iou >= 0.15:
                        pairs.append((iou * 0.9, si, di))
        pairs.sort(reverse=True)
        assigned_s: set[int] = set()
        assigned_d: set[int] = set()
        out: list[np.ndarray | None] = [None] * len(student_boxes)
        for iou, si, di in pairs:
            if si in assigned_s or di in assigned_d:
                continue
            assigned_s.add(si)
            assigned_d.add(di)
            out[si] = _probs_from_det(dets[di], self._channel_of)
        return out


def _names_from_model(model) -> list[str]:
    """Read the class names out of a loaded detector, whichever way it stores them."""
    raw = getattr(model, "names", None)
    if raw is None:
        raw = getattr(getattr(model, "model", None), "names", None)
    if callable(raw):
        try:
            raw = raw()
        except Exception:
            raw = None
    if isinstance(raw, dict):
        return [str(raw[i]) for i in range(len(raw))]
    if raw:
        return [str(n) for n in raw]
    return []


def _probs_from_det(d: SCBDet, channel_of: dict[int, int] | None = None) -> np.ndarray | None:
    """Turn one detection into a posture probability vector.

    The channel is resolved by class NAME first, so a checkpoint that names
    its classes differently from the pipeline taxonomy still lands in the
    right slot. Only when the name is unknown does the checkpoint's own
    class index act as a fallback.
    """
    probs = np.zeros(len(SCB_BEHAVIOR_LABELS), np.float32)
    name = d.name
    if name in SCB_BEHAVIOR_LABELS:
        probs[SCB_BEHAVIOR_LABELS.index(name)] = float(d.conf)
    elif channel_of:
        channel = channel_of.get(int(d.cls))
        if channel is not None:
            probs[channel] = float(d.conf)
    else:
        c = d.cls
        if 0 <= c < len(SCB_BEHAVIOR_LABELS):
            probs[c] = float(d.conf)
    if probs.sum() <= 0:
        return None
    return probs / max(probs.sum(), 1e-6)


def _ensure_yolov7_path() -> None:
    p = str(YOLOV7_DIR)
    if YOLOV7_DIR.exists() and p not in sys.path:
        sys.path.insert(0, p)


def _load_yolov7(path: str, device: str):
    import torch

    if not YOLOV7_DIR.exists():
        raise FileNotFoundError(
            f"Need {YOLOV7_DIR}. Run: python scripts/setup_scb_l2cs.py"
        )
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load
    try:
        _ensure_yolov7_path()
        from models.experimental import attempt_load

        model = attempt_load(path, map_location=device)
        model.eval()
        return model
    finally:
        torch.load = _orig


def _is_ultralytics_ckpt(path: str) -> bool:
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError:
        return False
    except Exception:
        return False
    if not isinstance(ckpt, dict):
        return False
    if "train_args" in ckpt:
        return True
    model = ckpt.get("model")
    if isinstance(model, dict):
        return True
    return False


def _canon_name(name: str) -> str:
    n = name.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "handraising": "hand-raising",
        "hand-rise": "hand-raising",
        "raise-hand": "hand-raising",
        "read": "reading",
        "write": "writing",
        "using-a-phone": "using-phone",
        "phone": "using-phone",
        "bow-head": "bowing-head",
        "bowhead": "bowing-head",
        "bowing-head": "bowing-head",
        "bowing-the-head": "bowing-head",
        "leaning-on-the-desk": "leaning-over-table",
        "leaning-over-the-table": "leaning-over-table",
        "leaning-over-table": "leaning-over-table",
        "leaning-on-table": "leaning-over-table",
    }
    return aliases.get(n, n)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def scb_attention_override(
    name: str,
    *,
    cues: dict[str, bool] | None = None,
    conf: float = 1.0,
) -> tuple[str, float, str] | None:
    """SCB sets the band only when pose/context agrees (multi-angle safe)."""
    cues = cues or {}
    if cues.get("phone") or cues.get("sleeping") or cues.get("head_on_table") or cues.get("look_left") or cues.get("look_right"):
        return None
    if cues.get("standing"):
        if name in {"reading", "writing", "bowing-head", "leaning-over-table"}:
            return None

    if name == "hand-raising":
        if cues.get("chin_rest") or cues.get("off_lesson") or cues.get("glance_left") or cues.get("glance_right"):
            return None
        if cues.get("hand_raised") or conf >= 0.7:
            return "HIGH", 93.0, "hand raised"
        return None

    if name in {"reading", "writing"}:
        if conf < 0.60:
            return None
        if cues.get("chin_rest") or cues.get("off_lesson"):
            return None
        if (cues.get("glance_left") or cues.get("glance_right")) and not cues.get("reading_tilt"):
            return None
        if cues.get("elevated_phone_grip") and name != "writing":
            return None
        desk_work = bool(cues.get("reading_tilt") and (cues.get("writing") or name == "writing"))
        # Side / front mix: reading needs a clear head-down cue.
        if name == "reading":
            desk_work = bool(cues.get("reading_tilt") or cues.get("writing"))
        watching = cues.get("facing_lesson") and cues.get("head_up") and cues.get("upright")
        if watching and not cues.get("reading_tilt"):
            return None
        if name == "reading" and not cues.get("reading_tilt") and conf < 0.80:
            return None
        cue = "reading" if name == "reading" else "writing notes"
        if desk_work or conf >= 0.80:
            return "HIGH", 90.0, cue
        return None

    if name == "bowing-head":
        if cues.get("still") and (cues.get("on_desk") or cues.get("collapsed")):
            return "LOW", 22.0, "sleeping / head on desk"
        if cues.get("reading_tilt") or cues.get("writing"):
            return "HIGH", 84.0, "reading / head to desk"
        return None

    if name == "using-phone":
        if cues.get("phone") or (conf >= 0.40 and not cues.get("chin_rest") and not cues.get("writing") and not cues.get("standing")):
            return "LOW", 24.0, "using phone"
        return None
    if name == "leaning-over-table":
        if cues.get("head_on_table"):
            return "LOW", 20.0, "head on table"
        return None
    return None
