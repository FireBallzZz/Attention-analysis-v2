"""Stage 2: one box per visible person, with no class-size cap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    bbox_xyxy: np.ndarray
    keypoints: np.ndarray  # (17, 3)
    conf: float
    track_id: int


class DetectorTracker:
    def __init__(
        self,
        weights: str = "yolo11n-pose.pt",
        conf: float = 0.08,
        iou: float = 0.5,
        device: str = "cpu",
        max_age: int = 30,
        n_init: int = 1,
        imgsz: int = 960,
        tiles: bool = True,
        person_weights: str = "yolo11n.pt",
        min_det_conf: float = 0.25,
        require_face: bool = True,
        max_people: int | None = None,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self._yolo = None
        self._person = None
        self.weights = weights
        self.person_weights = person_weights
        self.device = device
        self.max_age = max_age
        self.n_init = n_init
        self.imgsz = imgsz
        self.tiles = tiles
        # Drop empty wall/monitor boxes: need a face and a clearer detection.
        self.min_det_conf = float(min_det_conf)
        self.require_face = bool(require_face)
        self.max_people = max_people
        self._next_id = 1
        self._prev: list[tuple[int, np.ndarray]] = []
        self._instructor: tuple[float, float] | None = None
        self._instructor_misses = 0
        self._yunet = None
        self._yunet_ready = False

    def _lazy_load(self) -> None:
        if self._yolo is not None:
            return
        from ultralytics import YOLO

        self._yolo = YOLO(self.weights)
        if self.person_weights:
            try:
                self._person = YOLO(self.person_weights)
            except Exception:
                self._person = None

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """One box per visible person. The count is whoever is in this frame."""
        self._lazy_load()
        packed = self._detect_people(frame)
        frame_h = int(frame.shape[0])
        packed, self._instructor, self._instructor_misses = _drop_instructors(
            packed, frame_h, self._instructor, self._instructor_misses
        )
        if not packed:
            self._prev = []
            return []

        pairs: list[tuple[float, int, int]] = []
        for di, (box, _conf, _k) in enumerate(packed):
            for pi, (_tid, prev_box) in enumerate(self._prev):
                score = _track_score(box, prev_box)
                if score >= 0.35:
                    pairs.append((score, di, pi))
        pairs.sort(reverse=True)
        taken_det: set[int] = set()
        used_prev: set[int] = set()
        assigned: list[tuple[int, np.ndarray, float, np.ndarray]] = []
        for _score, di, pi in pairs:
            if di in taken_det or pi in used_prev:
                continue
            taken_det.add(di)
            used_prev.add(pi)
            box, conf, kpts = packed[di]
            assigned.append((self._prev[pi][0], box, conf, kpts))

        for di, (box, conf, kpts) in enumerate(packed):
            if di in taken_det:
                continue
            assigned.append((self._next_id, box, conf, kpts))
            self._next_id += 1

        self._prev = [(tid, box.copy()) for tid, box, _c, _k in assigned]
        dets: list[Detection] = []
        for tid, box, conf, kpts in assigned:
            k = kpts
            if k.ndim == 2 and k.shape[1] == 2:
                conf_col = np.ones((k.shape[0], 1), np.float32)
                k = np.concatenate([k, conf_col], axis=1)
            dets.append(
                Detection(
                    bbox_xyxy=box.astype(np.float32),
                    keypoints=k.astype(np.float32),
                    conf=float(conf),
                    track_id=int(tid),
                )
            )
        return dets

    def _detect_people(self, frame: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        h, w = frame.shape[:2]
        pose_hits = self._collect_pose(frame)
        people = _cluster_by_head(pose_hits, w)
        # Once the pose model has the class, person-only boxes are a second
        # mark on students who already have a face.
        if self._person is not None and len(people) < 12:
            people = _add_unmatched_people(
                people, self._collect_person_boxes(frame), w, h
            )
        people = _add_unmatched_faces(people, self._collect_faces(frame), w, h)
        people = _suppress_duplicate_boxes(people, w)
        people = self._keep_real_students(people)
        people = _drop_ceiling_specks(people, h, w)
        return _drop_tiny_near_face(people, w)

    def _keep_real_students(
        self, people: list[tuple[np.ndarray, float, np.ndarray]]
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Count only people with a visible face. Cap at the known class size."""
        kept: list[tuple[np.ndarray, float, np.ndarray]] = []
        for box, conf, kpts in people:
            if conf < self.min_det_conf:
                continue
            if self.require_face and not _face_confident(kpts):
                continue
            kept.append((box, conf, kpts))
        kept.sort(key=lambda item: item[1], reverse=True)
        if self.max_people is not None and len(kept) > self.max_people:
            kept = kept[: int(self.max_people)]
        return kept

    def count_students(self, frame: np.ndarray) -> int:
        """How many real students are in this frame (before the class-size cap)."""
        previous = self.max_people
        self.max_people = None
        n = len(self._detect_people(frame))
        self.max_people = previous
        return n

    def _regions(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        regions = [(0, 0, w, h)]
        if not self.tiles:
            return regions
        if w >= 1800:
            # Three columns and three rows. Back-row faces are too small in a
            # 2x2 crop of a 2560-wide classroom.
            xs = (0, int(w * 0.28), int(w * 0.56))
            xe = (int(w * 0.48), int(w * 0.76), w)
            ys = (0, int(h * 0.26), int(h * 0.50))
            ye = (int(h * 0.50), int(h * 0.78), h)
            for y0, y1 in zip(ys, ye):
                for x0, x1 in zip(xs, xe):
                    regions.append((x0, y0, x1, y1))
        elif w >= 1000:
            for y0, y1 in ((0, int(h * 0.72)), (int(h * 0.28), h)):
                for x0, x1 in ((0, int(w * 0.68)), (int(w * 0.32), w)):
                    regions.append((x0, y0, x1, y1))
        elif w >= 640:
            regions.append((0, 0, int(w * 0.62), h))
            regions.append((int(w * 0.38), 0, w, h))
        return regions

    def _collect_pose(self, frame: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Full frame first. Tiles only add a person the full pass missed."""
        h, w = frame.shape[:2]
        regions = self._regions(h, w)
        full = _shift_pose(
            self._yolo.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )[0],
            0,
            0,
        )
        if not self.tiles or len(regions) <= 1:
            return full

        people = _cluster_by_head(full, w)
        heads = [_head_point(box, kpts, w) for box, _c, kpts in people]
        # Same threshold as the full frame. A higher tile threshold drops
        # small students in the back rows.
        tile_conf = self.conf
        for x0, y0, x1, y1 in regions[1:]:
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            result = self._yolo.predict(
                crop,
                conf=tile_conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )[0]
            for box, conf, kpts in _shift_pose(result, x0, y0):
                hx, hy, hs = _head_point(box, kpts, w)
                if any(_heads_match(hx, hy, hs, px, py, ps) for px, py, ps in heads):
                    continue
                if any(
                    float(person[0]) <= hx <= float(person[2])
                    and float(person[1]) <= hy <= float(person[3])
                    for person, _c, _k in people
                ):
                    continue
                people.append((box, conf, kpts))
                heads.append((hx, hy, hs))
        return people

    def _collect_faces(self, frame: np.ndarray) -> list[tuple[np.ndarray, float, np.ndarray]]:
        """Faces the pose model misses, especially students looking down."""
        import cv2

        detector = self._lazy_yunet()
        if detector is None:
            return []
        h, w = frame.shape[:2]
        found: list[tuple[np.ndarray, float, np.ndarray]] = []
        regions = [(0, 0, w, h)]
        if w >= 1400:
            regions.extend(
                (
                    (0, int(h * 0.45), int(w * 0.55), h),
                    (int(w * 0.45), int(h * 0.45), w, h),
                    (0, int(h * 0.15), int(w * 0.55), int(h * 0.7)),
                    (int(w * 0.45), int(h * 0.15), w, int(h * 0.7)),
                )
            )
        for x0, y0, x1, y1 in regions:
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            detector.setInputSize((crop.shape[1], crop.shape[0]))
            try:
                _, faces = detector.detect(crop)
            except cv2.error:
                continue
            if faces is None:
                continue
            for face in faces:
                score = float(face[-1])
                if score < 0.45:
                    continue
                box, kpts = _face_to_student(face, x0, y0, w, h)
                found.append((box, score, kpts))
        return found

    def _lazy_yunet(self):
        if self._yunet_ready:
            return self._yunet
        self._yunet_ready = True
        path = _yunet_weights()
        if path is None:
            return None
        try:
            import cv2
        except ImportError:
            return None
        self._yunet = cv2.FaceDetectorYN.create(path, "", (320, 320), 0.45, 0.3, 5000)
        return self._yunet

    def _collect_person_boxes(self, frame: np.ndarray) -> list[tuple[np.ndarray, float]]:
        h, w = frame.shape[:2]
        found: list[tuple[np.ndarray, float]] = []
        for x0, y0, x1, y1 in self._regions(h, w):
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            result = self._person.predict(
                crop,
                conf=max(self.conf, 0.12),
                iou=self.iou,
                imgsz=self.imgsz,
                classes=[0],
                verbose=False,
                device=self.device,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue
            xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = result.boxes.conf.cpu().numpy()
            xyxy[:, [0, 2]] += x0
            xyxy[:, [1, 3]] += y0
            for box, conf in zip(xyxy, confs):
                if _box_area(box) < 24 * 36:
                    continue
                found.append((box, float(conf)))
        return found


def _shift_pose(result, x0: int, y0: int) -> list[tuple[np.ndarray, float, np.ndarray]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    confs = result.boxes.conf.cpu().numpy()
    if result.keypoints is not None:
        kpts = result.keypoints.data.cpu().numpy().astype(np.float32)
    else:
        kpts = np.zeros((len(xyxy), 17, 3), np.float32)
    xyxy[:, [0, 2]] += x0
    xyxy[:, [1, 3]] += y0
    kpts[:, :, 0] += x0
    kpts[:, :, 1] += y0
    out = []
    for box, conf, k in zip(xyxy, confs, kpts):
        if _box_area(box) < 18 * 28:
            continue
        out.append((box, float(conf), k))
    return out


def _cluster_by_head(
    found: list[tuple[np.ndarray, float, np.ndarray]],
    frame_w: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Merge tile duplicates that share a head. Keep neighbors whose heads differ."""
    if not found:
        return []
    heads = [_head_point(box, kpts, frame_w) for box, _c, kpts in found]
    order = sorted(range(len(found)), key=lambda i: found[i][1], reverse=True)
    used: set[int] = set()
    kept: list[tuple[np.ndarray, float, np.ndarray]] = []
    for i in order:
        if i in used:
            continue
        group = [i]
        used.add(i)
        hx, hy, hs = heads[i]
        for j in order:
            if j in used:
                continue
            jx, jy, js = heads[j]
            if _heads_match(hx, hy, hs, jx, jy, js):
                group.append(j)
                used.add(j)
        best = max(group, key=lambda idx: found[idx][1])
        _box, conf, kpts = found[best]
        merged = kpts.copy()
        for idx in group:
            other = found[idx][2]
            better = other[:, 2] > merged[:, 2]
            merged[better] = other[better]
        head = _head_point(_box, merged, frame_w)
        kept.append((_tight_box(_box, merged, head), conf, merged))
    return kept


def _add_unmatched_people(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    person_boxes: list[tuple[np.ndarray, float]],
    frame_w: int,
    frame_h: int = 0,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Add a person the pose model missed. Do not add a second box on someone already marked."""
    heads = [_head_point(box, kpts, frame_w) for box, _c, kpts in people]
    for box, conf in sorted(person_boxes, key=lambda item: item[1], reverse=True):
        if frame_h > 0 and not _person_shaped(box, frame_w, frame_h):
            continue
        if conf < 0.18:
            continue
        hx, hy, hs = _person_head(box, frame_w)
        if any(_heads_match(hx, hy, hs, px, py, ps) for px, py, ps in heads):
            continue
        # Skip when this detection is a second box on someone already marked.
        covered = False
        for person_box, _c, _k in people:
            if person_box[0] <= hx <= person_box[2] and person_box[1] <= hy <= person_box[3]:
                covered = True
                break
        if covered:
            continue
        kpts = _seed_face(box)
        tight = _tight_box(box, kpts, (hx, hy, hs))
        if any(
            _duplicate_of(tight, kpts, (hx, hy, hs), person_box, person_kpts, head, frame_w)
            for (person_box, _c, person_kpts), head in zip(people, heads)
        ):
            continue
        people.append((tight, conf, kpts))
        heads.append((hx, hy, hs))
    return people


def _drop_instructors(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    frame_h: int,
    remembered: tuple[float, float] | None,
    misses: int,
) -> tuple[list[tuple[np.ndarray, float, np.ndarray]], tuple[float, float] | None, int]:
    """Remove a standing instructor at the front. Seated students stay marked.

    The instructor is the person whose legs are extended and whose head sits
    clearly above the seated rows. At most two people match. A recent instructor
    head is kept unmarked for a short time if the legs are hidden for a frame.
    """
    if len(people) < 4:
        return people, remembered, misses + 1

    heads = [_head_top(box, kpts) for box, _conf, kpts in people]
    band = sorted(head[1] for head in heads)[max(0, int(0.2 * (len(heads) - 1)))]
    margin = max(50.0, 0.05 * frame_h)
    standing_above: list[int] = []
    for index, ((box, _conf, kpts), head) in enumerate(zip(people, heads)):
        if _legs_extended(kpts) and head[1] < band - margin:
            standing_above.append(index)

    # More than two matches means the pose cue is not separating an instructor.
    if len(standing_above) > 2:
        standing_above = []

    drop: set[int] = set(standing_above)
    if remembered is not None and len(standing_above) <= 2:
        reach = max(80.0, 0.06 * frame_h)
        for index, head in enumerate(heads):
            dist = ((head[0] - remembered[0]) ** 2 + (head[1] - remembered[1]) ** 2) ** 0.5
            if dist <= reach and head[1] < band - margin * 0.5:
                drop.add(index)

    if not drop:
        misses += 1
        if misses > 12:
            remembered = None
        return people, remembered, misses

    kept = [person for index, person in enumerate(people) if index not in drop]
    chosen = min(drop, key=lambda index: heads[index][1])
    remembered = heads[chosen]
    return kept, remembered, 0


def _head_top(box: np.ndarray, kpts: np.ndarray) -> tuple[float, float]:
    xs = [(float(box[0]) + float(box[2])) / 2.0]
    ys = [float(box[1])]
    for index in (0, 1, 2, 3, 4):
        if _kp_ok(kpts, index, 0.25):
            xs.append(float(kpts[index, 0]))
            ys.append(float(kpts[index, 1]))
    return sum(xs) / len(xs), min(ys)


def _legs_extended(kpts: np.ndarray) -> bool:
    """True when the thighs run downward, which seated desk rows do not show."""
    if not (_kp_ok(kpts, 11, 0.4) and _kp_ok(kpts, 12, 0.4)):
        return False
    knee_ys = [float(kpts[index, 1]) for index in (13, 14) if _kp_ok(kpts, index, 0.45)]
    if not knee_ys:
        return False
    hip = (float(kpts[11, 1]) + float(kpts[12, 1])) / 2.0
    knee = sum(knee_ys) / len(knee_ys)
    thigh = knee - hip
    if thigh < 24.0:
        return False
    if _kp_ok(kpts, 0, 0.25):
        torso = hip - float(kpts[0, 1])
    elif _kp_ok(kpts, 5, 0.3) and _kp_ok(kpts, 6, 0.3):
        shoulder = (float(kpts[5, 1]) + float(kpts[6, 1])) / 2.0
        torso = hip - shoulder
    else:
        return False
    return thigh >= 0.45 * max(torso, 8.0)


def _kp_ok(kpts: np.ndarray, index: int, thresh: float = 0.25) -> bool:
    return (
        kpts is not None
        and kpts.ndim == 2
        and kpts.shape[0] > index
        and kpts.shape[1] >= 3
        and float(kpts[index, 2]) > thresh
    )


def _head_point(box: np.ndarray, kpts: np.ndarray, frame_w: int) -> tuple[float, float, float]:
    """Head center and head size in pixels. Size comes from the face, not the body box."""
    cap = max(28.0, 0.07 * frame_w)

    def capped(size: float) -> float:
        return float(min(max(size, 12.0), cap))

    if _kp_ok(kpts, 1) and _kp_ok(kpts, 2):
        dist = _dist(kpts[1], kpts[2])
        cx = (float(kpts[1, 0]) + float(kpts[2, 0])) / 2.0
        cy = (float(kpts[1, 1]) + float(kpts[2, 1])) / 2.0
        return cx, cy, capped(dist * 2.6)
    if _kp_ok(kpts, 3) and _kp_ok(kpts, 4):
        dist = _dist(kpts[3], kpts[4])
        cx = (float(kpts[3, 0]) + float(kpts[4, 0])) / 2.0
        cy = (float(kpts[3, 1]) + float(kpts[4, 1])) / 2.0
        return cx, cy, capped(dist * 1.2)
    if _kp_ok(kpts, 0):
        size = 0.34 * max(float(box[2] - box[0]), 12.0)
        if _kp_ok(kpts, 5) and _kp_ok(kpts, 6):
            size = 0.42 * _dist(kpts[5], kpts[6])
        return float(kpts[0, 0]), float(kpts[0, 1]), capped(size)
    if _kp_ok(kpts, 5) and _kp_ok(kpts, 6):
        span = _dist(kpts[5], kpts[6])
        cx = (float(kpts[5, 0]) + float(kpts[6, 0])) / 2.0
        cy = min(float(kpts[5, 1]), float(kpts[6, 1])) - 0.35 * span
        return cx, cy, capped(0.45 * span)
    bw = max(float(box[2] - box[0]), 12.0)
    bh = max(float(box[3] - box[1]), 12.0)
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        float(box[1]) + 0.2 * bh,
        capped(0.36 * bw),
    )


def _yunet_weights() -> str | None:
    name = "face_detection_yunet_2023mar.onnx"
    candidates = [
        Path("checkpoints") / name,
        Path(__file__).resolve().parents[3] / "checkpoints" / name,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _face_to_student(
    face: np.ndarray,
    x0: int,
    y0: int,
    frame_w: int,
    frame_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a YuNet face into an upper-body box and nose/eye keypoints."""
    x, y, fw, fh = (float(v) for v in face[:4])
    x += x0
    y += y0
    cx = x + fw / 2.0
    box = np.array(
        [
            max(0.0, cx - fw * 0.95),
            max(0.0, y - 0.2 * fh),
            min(float(frame_w - 1), cx + fw * 0.95),
            min(float(frame_h - 1), y + fh * 2.3),
        ],
        np.float32,
    )
    kpts = np.zeros((17, 3), np.float32)
    # YuNet order: right eye, left eye, nose, right mouth, left mouth.
    reye = (float(face[4]) + x0, float(face[5]) + y0)
    leye = (float(face[6]) + x0, float(face[7]) + y0)
    nose = (float(face[8]) + x0, float(face[9]) + y0)
    kpts[0] = [nose[0], nose[1], 0.9]
    kpts[1] = [leye[0], leye[1], 0.9]
    kpts[2] = [reye[0], reye[1], 0.9]
    return box, kpts


def _add_unmatched_faces(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    faces: list[tuple[np.ndarray, float, np.ndarray]],
    frame_w: int,
    frame_h: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Keep a detected face when no pose box already covers that student."""
    heads = [_head_point(box, kpts, frame_w) for box, _c, kpts in people]
    for box, conf, kpts in sorted(faces, key=lambda item: item[1], reverse=True):
        hx, hy, hs = _head_point(box, kpts, frame_w)
        # A second detector often sits on a student the pose model already marked.
        if any(_same_student(hx, hy, hs, px, py, ps, frame_w) for px, py, ps in heads):
            continue
        people.append((box, conf, kpts))
        heads.append((hx, hy, hs))
    return people


def _drop_ceiling_specks(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    frame_h: int,
    frame_w: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Drop a tiny box floating above the class, such as a doorway mark."""
    if len(people) < 8:
        return people
    heads = [_head_point(box, kpts, frame_w) for box, _c, kpts in people]
    band = sorted(head[1] for head in heads)[max(0, int(0.2 * (len(heads) - 1)))]
    margin = 0.10 * frame_h
    kept: list[tuple[np.ndarray, float, np.ndarray]] = []
    for person, head in zip(people, heads):
        _box, conf, _kpts = person
        if head[1] < band - margin and head[2] < 0.03 * frame_h and conf < 0.25:
            continue
        kept.append(person)
    return kept


def _drop_tiny_near_face(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    frame_w: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """A tiny second mark sitting on a real face is not another student."""
    if len(people) < 2:
        return people
    heads = [_head_point(box, kpts, frame_w) for box, _c, kpts in people]
    drop: set[int] = set()
    for i, hi in enumerate(heads):
        if hi[2] >= 28.0:
            continue
        for j, hj in enumerate(heads):
            if i == j or hj[2] < 36.0:
                continue
            dist = ((hi[0] - hj[0]) ** 2 + (hi[1] - hj[1]) ** 2) ** 0.5
            if dist < max(36.0, 0.55 * hj[2]):
                drop.add(i)
                break
    return [person for index, person in enumerate(people) if index not in drop]


def _person_shaped(box: np.ndarray, frame_w: int, frame_h: int) -> bool:
    """A seated student, not a speck and not a wall or monitor."""
    bw = float(box[2] - box[0])
    bh = float(box[3] - box[1])
    if bh < 0.04 * frame_h or bh > 0.5 * frame_h:
        return False
    if bw < 0.012 * frame_w or bw > 0.28 * frame_w:
        return False
    ratio = bw / max(bh, 1.0)
    return 0.22 <= ratio <= 1.05


def _seed_face(box: np.ndarray) -> np.ndarray:
    """Place a face on a person box so gaze can crop the head."""
    kpts = np.zeros((17, 3), np.float32)
    bw = float(box[2] - box[0])
    bh = float(box[3] - box[1])
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = float(box[1]) + 0.16 * bh
    eye = max(4.0, 0.11 * bw)
    kpts[0] = [cx, cy + 0.05 * bh, 0.55]
    kpts[1] = [cx - eye, cy, 0.55]
    kpts[2] = [cx + eye, cy, 0.55]
    return kpts


def _person_head(box: np.ndarray, frame_w: int) -> tuple[float, float, float]:
    bw = max(float(box[2] - box[0]), 12.0)
    bh = max(float(box[3] - box[1]), 12.0)
    cap = max(28.0, 0.07 * frame_w)
    return (
        (float(box[0]) + float(box[2])) / 2.0,
        float(box[1]) + 0.18 * bh,
        float(min(max(0.34 * bw, 12.0), cap)),
    )


def _same_student(
    ax: float,
    ay: float,
    a_size: float,
    bx: float,
    by: float,
    b_size: float,
    frame_w: int,
) -> bool:
    """True when two marks are the same child, including a second detector."""
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    small = min(a_size, b_size)
    dy = abs(ay - by)
    # Scale with the frame so a small clip does not merge the next student.
    near = max(16.0, 0.016 * frame_w)
    row = max(20.0, 0.026 * frame_w)
    row_dy = max(10.0, 0.011 * frame_w)
    if dist < max(near, 0.75 * small):
        return True
    return dy <= max(row_dy, 0.45 * small) and dist < max(row, 1.0 * small)


def _heads_match(ax: float, ay: float, a_size: float, bx: float, by: float, b_size: float) -> bool:
    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    small = min(a_size, b_size)
    dy = abs(ay - by)
    # Same row and almost the same face: a tile and a face detector on one student.
    if dy <= max(14.0, 0.4 * small) and dist < max(24.0, 0.85 * small):
        return True
    # Tile crops land a few pixels apart on the same face. A wider radius
    # merges the next student in a packed row.
    return dist < max(18.0, 0.62 * small)


def _suppress_duplicate_boxes(
    people: list[tuple[np.ndarray, float, np.ndarray]],
    frame_w: int,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Drop a second box on the same body after boxes have been tightened."""
    kept: list[tuple[np.ndarray, float, np.ndarray]] = []
    heads: list[tuple[float, float, float]] = []
    for box, conf, kpts in sorted(people, key=lambda item: item[1], reverse=True):
        hx, hy, hs = _head_point(box, kpts, frame_w)
        duplicate = False
        for (other, _conf, other_kpts), (ox, oy, os_) in zip(kept, heads):
            if _duplicate_of(
                box, kpts, (hx, hy, hs), other, other_kpts, (ox, oy, os_), frame_w
            ):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((box, conf, kpts))
        heads.append((hx, hy, hs))
    return kept


def _point_in_box(x: float, y: float, box: np.ndarray, pad: float = 0.0) -> bool:
    return (
        float(box[0]) - pad <= x <= float(box[2]) + pad
        and float(box[1]) - pad <= y <= float(box[3]) + pad
    )


def _duplicate_of(
    box: np.ndarray,
    kpts: np.ndarray,
    head: tuple[float, float, float],
    other: np.ndarray,
    other_kpts: np.ndarray,
    other_head: tuple[float, float, float],
    frame_w: int = 1920,
) -> bool:
    hx, hy, hs = head
    ox, oy, os_ = other_head
    dist = ((hx - ox) ** 2 + (hy - oy) ** 2) ** 0.5
    if _heads_match(hx, hy, hs, ox, oy, os_) or _same_student(
        hx, hy, hs, ox, oy, os_, frame_w
    ):
        return True
    # A student in the next row can overlap the box behind them.
    if _face_confident(kpts) and _face_confident(other_kpts):
        return False
    iou = _iou(box, other)
    iomin = _iomin(box, other)
    # A small box mostly inside a larger one is the same student counted twice.
    if iomin >= 0.5:
        return True
    if iomin >= 0.35 and (
        _point_in_box(hx, hy, other, pad=8.0) or _point_in_box(ox, oy, box, pad=8.0)
    ):
        return True
    overlap = iou >= 0.35 or iomin >= 0.45
    if overlap and dist < max(28.0, 0.7 * min(hs, os_)):
        return True
    dy = abs(hy - oy)
    if dy < 0.45 * max(hs, os_) and dist < 0.75 * max(hs, os_) and iomin >= 0.4:
        return True
    return _torso_under_face(box, kpts, head, other, other_kpts, other_head)


def _torso_under_face(
    box: np.ndarray,
    kpts: np.ndarray,
    head: tuple[float, float, float],
    other: np.ndarray,
    other_kpts: np.ndarray,
    other_head: tuple[float, float, float],
) -> bool:
    """A chest box with no face, directly under a real face, is the same person."""
    hx, hy, _hs = head
    ox, oy, _os = other_head
    dx = abs(hx - ox)
    width = max(float(box[2] - box[0]), float(other[2] - other[0]), 1.0)
    if dx >= 0.45 * width or not _stacked(box, other):
        return False
    if hy >= oy:
        lower_k, upper_k = kpts, other_kpts
    else:
        lower_k, upper_k = other_kpts, kpts
    return _face_confident(upper_k) and not _face_confident(lower_k)


def _face_confident(kpts: np.ndarray) -> bool:
    return _kp_ok(kpts, 0, 0.45) or (_kp_ok(kpts, 1, 0.35) and _kp_ok(kpts, 2, 0.35))


def _stacked(a: np.ndarray, b: np.ndarray) -> bool:
    top = max(float(a[1]), float(b[1]))
    bottom = min(float(a[3]), float(b[3]))
    gap = top - bottom
    smaller_h = min(float(a[3] - a[1]), float(b[3] - b[1]))
    return gap < 0.2 * max(smaller_h, 1.0)


def _tight_box(
    box: np.ndarray,
    kpts: np.ndarray,
    head: tuple[float, float, float],
) -> np.ndarray:
    """Upper-body box around the head so a neighbor behind is not covered."""
    cx, cy, size = head
    width = size * 1.9
    if _kp_ok(kpts, 5) and _kp_ok(kpts, 6):
        width = max(width, _dist(kpts[5], kpts[6]) * 1.05)
    top = cy - size * 0.9
    bottom = cy + size * 2.15
    left = cx - width / 2.0
    right = cx + width / 2.0
    # Keep the box inside the original detection so it cannot jump onto a neighbor.
    left = max(left, float(box[0]))
    right = min(right, float(box[2]))
    top = max(top, float(box[1]))
    bottom = min(bottom, float(box[3]))
    if right - left < 18:
        left, right = float(box[0]), float(box[2])
    if bottom - top < 24:
        top, bottom = float(box[1]), float(box[3])
    return np.array([left, top, right, bottom], np.float32)


def _iomin(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    small = min(_box_area(a), _box_area(b))
    return inter / small if small > 0 else 0.0


def _track_score(a: np.ndarray, b: np.ndarray) -> float:
    iou = _iou(a, b)
    cax = (float(a[0]) + float(a[2])) / 2.0
    cay = (float(a[1]) + float(a[3])) / 2.0
    cbx = (float(b[0]) + float(b[2])) / 2.0
    cby = (float(b[1]) + float(b[3])) / 2.0
    dist = ((cax - cbx) ** 2 + (cay - cby) ** 2) ** 0.5
    side = min(
        max(float(a[2] - a[0]), 1.0),
        max(float(a[3] - a[1]), 1.0),
        max(float(b[2] - b[0]), 1.0),
        max(float(b[3] - b[1]), 1.0),
    )
    if dist < 0.5 * side:
        return 1.0
    return iou


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2) ** 0.5)


def _box_area(box: np.ndarray) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = _box_area(a)
    area_b = _box_area(b)
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0
