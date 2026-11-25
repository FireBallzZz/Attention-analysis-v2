"""Defence dashboard server: the dashboard + "upload a video and analyse it".

    python launch_defence_dashboard.py            # http://127.0.0.1:8765, opens the browser
    python launch_defence_dashboard.py --no-browser --port 8765

Serves dashboard/defence/. Uploaded videos are analysed in a background
thread by the full pipeline (configs/pipeline.yaml, final model), one at a
time. Each finished analysis is saved in runs/defence_uploads/ and listed on
the dashboard's "Analyse a video" page, including after a restart.
Rebuild the demo data with:  python scripts/build_defence_dashboard.py
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402

DASH = ROOT / "dashboard" / "defence"
UPLOADS = ROOT / "runs" / "defence_uploads"
ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

_jobs: dict = {}
_jobs_lock = threading.Lock()
_worker_lock = threading.Lock()  # one GPU -> one analysis at a time
_processor = None


def processor():
    global _processor
    if _processor is None:
        from video_processor import VideoProcessor

        _processor = VideoProcessor()
    return _processor


def _run(job, src: Path, stem: str, title: str) -> None:
    from dashboard_payload import build_payload, clean_frame

    with _worker_lock:
        out_video = UPLOADS / f"{stem}_annotated.mp4"
        out_jsonl = UPLOADS / f"{stem}_scores.jsonl"
        processor().run(job, src, out_video, out_jsonl)
        if job.status != "done":
            return
        try:
            job.stage = "Building dashboard view"
            size = clean_frame(src, UPLOADS / f"{stem}_frame.jpg")
            payload = build_payload(
                key=stem, title=title, room="uploaded", labelled=False, jsonl=out_jsonl,
                video_url=f"uploads/{out_video.name}", poster_url=f"uploads/{stem}_frame.jpg", frame_size=size)
            payload["uploaded_at"] = time.strftime("%Y-%m-%d %H:%M")
            payload["processing"] = job.to_dict()["timing"] | job.to_dict()["video"]
            (UPLOADS / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")
            job.stage = "Complete"
        except Exception as exc:  # keep the annotated video even if the view fails
            job.status, job.stage, job.error = "error", "Failed", f"dashboard view: {exc}"


@app.get("/")
def index():
    return send_from_directory(DASH, "index.html")


@app.get("/<path:name>")
def static_files(name: str):
    if name.startswith("uploads/"):
        return send_from_directory(UPLOADS, name[len("uploads/"):])
    if (DASH / name).is_file():
        return send_from_directory(DASH, name)
    abort(404)


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


@app.post("/api/upload")
def upload():
    from video_processor import JobResult

    f = request.files.get("video")
    if f is None or not f.filename:
        return jsonify({"error": "No video file received."}), 400
    suffix = Path(f.filename).suffix.lower()
    if suffix not in ALLOWED:
        return jsonify({"error": f"Unsupported file type '{suffix}'. Use: {', '.join(sorted(ALLOWED))}"}), 400
    UPLOADS.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(f.filename).stem)[:50] or "video"
    stem = f"{job_id}_{safe}"
    src = UPLOADS / f"{stem}{suffix}"
    f.save(src)
    if src.stat().st_size == 0:
        src.unlink(missing_ok=True)
        return jsonify({"error": "The uploaded file is empty."}), 400
    job = JobResult(job_id=job_id, status="queued", stage="Queued", message="Waiting for the analysis worker",
                    input_name=f.filename, input_path=str(src), queued_at=time.time())
    with _jobs_lock:
        _jobs[job_id] = (job, stem)
    threading.Thread(target=_run, args=(job, src, stem, Path(f.filename).stem), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        entry = _jobs.get(job_id)
    if entry is None:
        return jsonify({"error": "Unknown job"}), 404
    job, stem = entry
    out = {"job_id": job_id, "status": job.status, "stage": job.stage, "progress": round(job.progress, 3),
           "error": job.error, "input_name": job.input_name}
    view = UPLOADS / f"{stem}.json"
    if job.status == "done" and view.exists():
        out["payload"] = json.loads(view.read_text(encoding="utf-8"))
    elif job.status == "done":
        out["status"] = "running"  # dashboard view still being built
    return jsonify(out)


@app.get("/api/uploads")
def uploads():
    items = []
    for p in sorted(UPLOADS.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return jsonify(items)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not (DASH / "data.js").exists():
        raise SystemExit("dashboard/defence/data.js missing - run: python scripts/build_defence_dashboard.py")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Defence dashboard: {url}   (keep this window open; Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
