"""Simple keyboard labelling tool for classroom attention (replaces CVAT).

You never draw boxes. Students are tracked automatically, then you watch one
student at a time in short looping segments and press one key per segment.

    python scripts/label_tool.py "runs/Test 5.mp4"
    python scripts/label_tool.py "runs/Test 5.mp4" --annotator B   # second person, for kappa

Keys
    1  HIGH (attentive)        2  LOW (not attentive)        0  Can't see / skip
    Enter      same label as the previous segment of this student
    A / Left   previous segment          D / Right  next segment (no change)
    W / Up     previous student          S / Down   next student
    Space      pause / play              H          show/hide all other boxes
    Q / Esc    save and quit

Progress is saved after every key press, so you can quit and resume any time.

Output (in Anno/):
    <video>_<annotator>_ground_truth.csv             frame,student_id,attention
    <video>_<annotator>_ground_truth_with_boxes.csv  + box per frame
    <video>_<annotator>_progress.json                resume state
    <video>_tracks.json                              cached student boxes
No model prediction is ever shown, so the labels stay independent of the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANNO = ROOT / "Anno"

# Two classes only. Written as Attentive / Non-Attentive so the existing
# loaders (High+Moderate -> engaged, Low -> disengaged) read them unchanged.
SCHEME = "binary_high_low"
LABELS = {1: "Attentive", 2: "Non-Attentive", 0: "Skip"}
SHORT = {1: "HIGH", 2: "LOW", 0: "SKIP"}
COLORS = {1: (80, 200, 80), 2: (60, 60, 230), 0: (150, 150, 150), None: (255, 255, 255)}
RULES = [
    "1 HIGH: looks at teacher/board/screen,",
    "  OR writes/reads lesson material (head down OK),",
    "  OR raises hand. Short glances away (< 2 s),",
    "  chin on hand, turning pages also = HIGH.",
    "2 LOW (lasting > ~2 s): head turned far away,",
    "  phone, head on desk / sleeping, chatting,",
    "  playing with objects, eyes closed > 1 s.",
    "0 CAN'T SEE: hidden / out of view / no student.",
    "Unsure -> HIGH. Only clear cases are LOW.",
]

# Arrow-key codes from cv2.waitKeyEx (Windows and Linux/GTK)
K_LEFT = {2424832, 65361, ord("a"), ord("A")}
K_RIGHT = {2555904, 65363, ord("d"), ord("D")}
K_UP = {2490368, 65362, ord("w"), ord("W")}
K_DOWN = {2621440, 65364, ord("s"), ord("S")}
K_ENTER = {13, 10}
K_QUIT = {27, ord("q"), ord("Q")}


# ----------------------------------------------------------------- tracking
def track_video(video: Path, cache: Path, min_seconds: float) -> dict:
    """Run YOLO11-Pose + ByteTrack once and cache per-frame boxes."""
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
        print(f"Using cached tracks: {cache.name} ({len(data['tracks'])} students)")
        return data

    from ultralytics import YOLO

    weights = ROOT / "yolo11n-pose.pt"
    model = YOLO(str(weights if weights.exists() else "yolo11n-pose.pt"))
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5) or 30.0
    n = int(cap.get(7))
    imgsz = int(min(1920, max(1280, round(max(w, h) / 32) * 32)))
    print(f"Tracking students in {video.name} ({w}x{h}, {n} frames, imgsz={imgsz}) ...")

    raw: dict[int, dict[int, list[float]]] = defaultdict(dict)
    f = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=imgsz,
                          conf=0.20, verbose=False)[0]
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int)):
                raw[int(tid)][f] = [round(float(v), 1) for v in box]
        f += 1
        if f % 60 == 0:
            print(f"  frame {f}/{n}  ({time.time() - t0:.0f}s)")
    cap.release()

    n_raw = len(raw)
    raw = clean_tracks(raw, gap=int(3 * fps))
    # Drop very short tracks (flicker), renumber left-to-right, front-to-back.
    min_frames = int(min_seconds * fps)
    kept = {t: fr for t, fr in raw.items() if len(fr) >= min_frames}

    def centre(fr):
        b = np.array(list(fr.values()))
        return ((b[:, 0] + b[:, 2]) / 2).mean(), ((b[:, 1] + b[:, 3]) / 2).mean()

    order = sorted(kept, key=lambda t: (round(centre(kept[t])[1] / (h / 6)), centre(kept[t])[0]))
    tracks = {str(i): {str(fr): b for fr, b in kept[t].items()} for i, t in enumerate(order)}
    data = {"video": video.name, "width": w, "height": h, "fps": fps, "frames": f, "tracks": tracks}
    cache.write_text(json.dumps(data), encoding="utf-8")
    print(f"Kept {len(tracks)} students ({n_raw} raw tracks after merging broken/duplicate "
          f"tracks and dropping ones shorter than {min_seconds}s)")
    return data


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def clean_tracks(raw: dict, gap: int) -> dict:
    """Students sit still, so: (1) drop a track that duplicates another one
    (same person, overlapping in time, boxes mostly overlapping), and
    (2) join a track that ends with one that starts nearby shortly after."""
    tr = {t: dict(fr) for t, fr in raw.items()}

    # 1. duplicates: keep the longer track
    changed = True
    while changed:
        changed = False
        ids = sorted(tr, key=lambda t: -len(tr[t]))
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                common = set(tr[a]) & set(tr[b])
                if len(common) >= 0.5 * len(tr[b]) and np.mean(
                        [_iou(tr[a][f], tr[b][f]) for f in common]) > 0.5:
                    for f, box in tr[b].items():
                        tr[a].setdefault(f, box)
                    del tr[b]
                    changed = True
                    break
            if changed:
                break

    # 2. broken tracks: join end-to-start when close in time and place
    changed = True
    while changed:
        changed = False
        ids = sorted(tr, key=lambda t: min(tr[t]))
        for a in ids:
            end_a = max(tr[a])
            best, best_score = None, 0.0
            for b in ids:
                if b == a:
                    continue
                start_b = min(tr[b])
                if not (end_a - 5 <= start_b <= end_a + gap):
                    continue
                if len(set(tr[a]) & set(tr[b])) > 5:
                    continue
                score = _iou(tr[a][end_a], tr[b][start_b])
                if score > 0.3 and score > best_score:
                    best, best_score = b, score
            if best is not None:
                for f, box in tr[best].items():
                    tr[a].setdefault(f, box)
                del tr[best]
                changed = True
                break
    return tr


# ----------------------------------------------------------------- tool
class Labeller:
    def __init__(self, video: Path, data: dict, seg_seconds: float, annotator: str, max_w: int, max_h: int):
        self.video = video
        self.data = data
        self.fps = data["fps"]
        self.n_frames = data["frames"]
        self.seg_len = max(1, int(round(seg_seconds * self.fps)))
        self.n_segs = int(np.ceil(self.n_frames / self.seg_len))
        self.tracks = {int(k): {int(f): np.array(b) for f, b in v.items()} for k, v in data["tracks"].items()}
        self.ids = sorted(self.tracks)
        # A student "exists" in a segment if visible in >= 30% of its frames.
        self.segs_of = {
            s: [g for g in range(self.n_segs)
                if sum(f in self.tracks[s] for f in self.frames_of(g)) >= 0.3 * len(self.frames_of(g))]
            for s in self.ids
        }
        self.ids = [s for s in self.ids if self.segs_of[s]]
        stem = video.stem.replace(" ", "_")
        self.prefix = ANNO / f"{stem}_{annotator}"
        self.progress = Path(f"{self.prefix}_progress.json")
        self.labels: dict[tuple[int, int], int] = {}
        if self.progress.exists():
            saved = json.loads(self.progress.read_text(encoding="utf-8"))
            if saved.get("scheme") != SCHEME:
                sys.exit(f"{self.progress.name} was made with the old 3-label version. "
                         f"Rename or delete it to start again with HIGH/LOW.")
            self.labels = {(int(k.split(":")[0]), int(k.split(":")[1])): v for k, v in saved["labels"].items()}
            print(f"Resuming: {len(self.labels)} segment labels loaded")
        self.si, self.gi = 0, 0
        self.jump_to_first_unlabelled()
        self.scale = min(max_w / data["width"], max_h / data["height"], 1.6)
        self.cap = cv2.VideoCapture(str(video))
        self.cache: dict[int, list[np.ndarray]] = {}
        self.show_others = True

    def frames_of(self, g: int) -> range:
        return range(g * self.seg_len, min((g + 1) * self.seg_len, self.n_frames))

    @property
    def sid(self) -> int:
        return self.ids[self.si]

    @property
    def seg(self) -> int:
        return self.segs_of[self.sid][self.gi]

    def total_items(self) -> int:
        return sum(len(v) for v in self.segs_of.values())

    def jump_to_first_unlabelled(self) -> None:
        for i, s in enumerate(self.ids):
            for j, g in enumerate(self.segs_of[s]):
                if (s, g) not in self.labels:
                    self.si, self.gi = i, j
                    return

    def segment_frames(self, g: int) -> list[np.ndarray]:
        if g not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, g * self.seg_len)
            frames = []
            for _ in self.frames_of(g):
                ok, fr = self.cap.read()
                if not ok:
                    break
                frames.append(fr)
            self.cache[g] = frames
        return self.cache[g]

    def box_at(self, s: int, f: int):
        t = self.tracks[s]
        if f in t:
            return t[f]
        near = [k for k in t if abs(k - f) <= self.seg_len]
        return t[min(near, key=lambda k: abs(k - f))] if near else None

    # -------------------------------------------------------- drawing
    def render(self, k: int) -> np.ndarray:
        g = self.seg
        frames = self.segment_frames(g)
        if not frames:
            return np.zeros((400, 800, 3), np.uint8)
        k %= len(frames)
        f = g * self.seg_len + k
        img = frames[k].copy()
        cur = self.box_at(self.sid, f)

        dim = (img * 0.45).astype(np.uint8)
        if cur is not None:
            x1, y1, x2, y2 = cur.astype(int)
            dim[max(0, y1):y2, max(0, x1):x2] = img[max(0, y1):y2, max(0, x1):x2]
        img = dim
        th = max(1, int(round(2 / self.scale)))
        if self.show_others:
            for s in self.ids:
                if s == self.sid:
                    continue
                b = self.box_at(s, f)
                if b is not None:
                    x1, y1, x2, y2 = b.astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (110, 110, 110), max(1, th // 2))
        lab = self.labels.get((self.sid, g))
        if cur is not None:
            x1, y1, x2, y2 = cur.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[lab] if lab is not None else (255, 255, 0), th * 2)
            cv2.putText(img, f"#{self.sid}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 / self.scale * 1.2, (255, 255, 0), th)
        view = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)

        # Zoomed crop of the student
        panel_w = 380
        H = view.shape[0]
        crop_panel = np.zeros((max(H, 640), panel_w, 3), np.uint8)
        if cur is not None:
            x1, y1, x2, y2 = cur
            cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * 1.6, (y2 - y1) * 1.4
            a, b = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            c, d = int(min(frames[k].shape[1], cx + bw / 2)), int(min(frames[k].shape[0], cy + bh / 2))
            crop = frames[k][b:d, a:c]
            if crop.size:
                zs = min((panel_w - 20) / crop.shape[1], 260 / crop.shape[0])
                crop = cv2.resize(crop, None, fx=zs, fy=zs, interpolation=cv2.INTER_CUBIC)
                crop_panel[10:10 + crop.shape[0], 10:10 + crop.shape[1]] = crop

        y = 290
        def put(text, color=(230, 230, 230), scale=0.5, bold=1):
            nonlocal y
            cv2.putText(crop_panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, bold)
            y += int(22 * scale / 0.5)

        done = len(self.labels)
        put(f"Student #{self.sid}  ({self.si + 1}/{len(self.ids)})", (255, 255, 0), 0.6, 2)
        t0, t1 = g * self.seg_len / self.fps, min((g + 1) * self.seg_len, self.n_frames) / self.fps
        put(f"Segment {self.gi + 1}/{len(self.segs_of[self.sid])}   {t0:.1f}s - {t1:.1f}s")
        put(f"Label: {SHORT[lab] if lab is not None else '-- not set --'}", COLORS[lab], 0.6, 2)
        put(f"Progress: {done}/{self.total_items()}")
        y += 6
        for r in RULES:
            put(r, (200, 200, 200), 0.42)
        y += 4
        put("Enter=same as before  A/D=seg  W/S=student", (160, 220, 160), 0.42)
        put("Space=pause  H=other boxes  Q=save+quit", (160, 220, 160), 0.42)

        if crop_panel.shape[0] > H:
            view = cv2.copyMakeBorder(view, 0, crop_panel.shape[0] - H, 0, 0, cv2.BORDER_CONSTANT)
        # Segment timeline for this student
        bar = np.full((26, view.shape[1] + panel_w, 3), 30, np.uint8)
        segs = self.segs_of[self.sid]
        wseg = bar.shape[1] / max(1, len(segs))
        for j, gg in enumerate(segs):
            c = COLORS[self.labels.get((self.sid, gg))] if (self.sid, gg) in self.labels else (70, 70, 70)
            cv2.rectangle(bar, (int(j * wseg) + 1, 3), (int((j + 1) * wseg) - 1, 23), c, -1)
            if j == self.gi:
                cv2.rectangle(bar, (int(j * wseg), 1), (int((j + 1) * wseg) - 1, 25), (255, 255, 255), 2)
        return np.vstack([np.hstack([view, crop_panel]), bar])

    # -------------------------------------------------------- navigation
    def next_item(self) -> None:
        if self.gi + 1 < len(self.segs_of[self.sid]):
            self.gi += 1
        elif self.si + 1 < len(self.ids):
            self.si, self.gi = self.si + 1, 0

    def prev_item(self) -> None:
        if self.gi > 0:
            self.gi -= 1
        elif self.si > 0:
            self.si -= 1
            self.gi = len(self.segs_of[self.sid]) - 1

    def set_label(self, v: int) -> None:
        self.labels[(self.sid, self.seg)] = v
        self.save_progress()
        self.next_item()

    def save_progress(self) -> None:
        self.progress.write_text(json.dumps({
            "video": self.video.name, "scheme": SCHEME, "segment_seconds": self.seg_len / self.fps,
            "labels": {f"{s}:{g}": v for (s, g), v in self.labels.items()},
        }, indent=0), encoding="utf-8")

    def export(self) -> None:
        rows = []
        for (s, g), v in sorted(self.labels.items()):
            if v == 0:
                continue
            for f in self.frames_of(g):
                b = self.tracks[s].get(f)
                if b is None:
                    continue
                rows.append((f, s, LABELS[v], *[round(float(x), 2) for x in b]))
        rows.sort()
        p1 = Path(f"{self.prefix}_ground_truth.csv")
        p2 = Path(f"{self.prefix}_ground_truth_with_boxes.csv")
        with p1.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "student_id", "attention"])
            w.writerows(r[:3] for r in rows)
        with p2.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "student_id", "attention", "xtl", "ytl", "xbr", "ybr"])
            w.writerows(rows)
        counts = {LABELS[k]: sum(1 for v in self.labels.values() if v == k) for k in LABELS}
        print(f"Saved {p1.name} and {p2.name} ({len(rows)} frame rows). Segment counts: {counts}")
        print(f"Labelled {len(self.labels)}/{self.total_items()} segments.")

    def run(self) -> None:
        win = f"Label: {self.video.name}  (Q = save and quit)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        k, paused, last = 0, False, time.time()
        delay = 1.0 / min(self.fps, 15.0)
        step = max(1, int(round(self.fps / 15.0)))
        while True:
            cv2.imshow(win, self.render(k))
            key = cv2.waitKeyEx(10)
            if not paused and time.time() - last >= delay:
                k += step
                last = time.time()
            if key == -1:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            before = (self.si, self.gi)
            if key in K_QUIT:
                break
            elif key in (ord("1"), ord("2"), ord("0")):
                self.set_label(int(chr(key)))
            elif key in K_ENTER:
                prev = None
                segs = self.segs_of[self.sid]
                for gg in reversed(segs[:self.gi]):
                    if (self.sid, gg) in self.labels:
                        prev = self.labels[(self.sid, gg)]
                        break
                if prev is not None:
                    self.set_label(prev)
            elif key in K_RIGHT:
                self.next_item()
            elif key in K_LEFT:
                self.prev_item()
            elif key in K_DOWN and self.si + 1 < len(self.ids):
                self.si, self.gi = self.si + 1, 0
            elif key in K_UP and self.si > 0:
                self.si, self.gi = self.si - 1, 0
            elif key == ord(" "):
                paused = not paused
            elif key in (ord("h"), ord("H")):
                self.show_others = not self.show_others
            if (self.si, self.gi) != before:
                k = 0
        cv2.destroyAllWindows()
        self.save_progress()
        self.export()


def main() -> None:
    ap = argparse.ArgumentParser(description="Keyboard attention labelling tool (no box drawing).")
    ap.add_argument("video", help="Video file, e.g. \"runs/Test 5.mp4\"")
    ap.add_argument("--annotator", default="A", help="Your initials; a second person uses another letter")
    ap.add_argument("--segment-seconds", type=float, default=2.0)
    ap.add_argument("--min-track-seconds", type=float, default=1.0, help="Ignore students visible less than this")
    ap.add_argument("--max-width", type=int, default=1150, help="Max video width on screen")
    ap.add_argument("--max-height", type=int, default=700)
    ap.add_argument("--retrack", action="store_true", help="Ignore cached tracks and track again")
    ap.add_argument("--snapshot", default="", help="Render the first screen to this PNG and exit (testing)")
    ap.add_argument("--export-only", action="store_true", help="Write the CSVs from saved progress and exit")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = (ROOT / video) if (ROOT / video).exists() else video.resolve()
    if not video.exists():
        sys.exit(f"Video not found: {args.video}")
    ANNO.mkdir(exist_ok=True)
    cache = ANNO / f"{video.stem.replace(' ', '_')}_tracks.json"
    if args.retrack and cache.exists():
        cache.unlink()
    data = track_video(video, cache, args.min_track_seconds)
    tool = Labeller(video, data, args.segment_seconds, args.annotator, args.max_width, args.max_height)
    print(f"{len(tool.ids)} students x segments = {tool.total_items()} decisions "
          f"(segments of {tool.seg_len / tool.fps:.1f}s)")
    if args.snapshot:
        cv2.imwrite(args.snapshot, tool.render(0))
        print(f"Wrote {args.snapshot}")
        return
    if args.export_only:
        tool.export()
        return
    tool.run()


if __name__ == "__main__":
    main()
