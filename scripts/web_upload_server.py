"""Flask server for the video-upload attention analysis dashboard.

Endpoints:
  GET  /                       the single-page upload UI
  POST /api/upload            accept a video, queue a job, return job id
  GET  /api/job/<job_id>      poll job status, progress and timings
  GET  /api/jobs              list recent jobs
  GET  /outputs/<name>        serve the annotated output video / scores
"""

from __future__ import annotations

import argparse
import shutil
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request, send_from_directory

from video_processor import JobResult, VideoProcessor

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_ROOT / "runs" / "web_uploads"
DASHBOARD_DIR = WORKSPACE_ROOT / "dashboard"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_processor = VideoProcessor()
_jobs: dict[str, JobResult] = {}
_jobs_lock = threading.Lock()

# Uploaded videos and their outputs are deleted after this many hours so the
# output directory does not grow without bound.
RETENTION_HOURS = 6


def _purge_old_outputs() -> None:
    """Delete uploaded files and outputs older than the retention window."""
    cutoff = time.time() - RETENTION_HOURS * 3600
    removed = 0
    for path in OUTPUT_DIR.glob("*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    if removed:
        print(f"[purge] removed {removed} expired upload(s)")


def _purge_loop() -> None:
    while True:
        time.sleep(600)
        try:
            _purge_old_outputs()
        except Exception:
            pass


def _run_job(job: JobResult, input_path: Path, out_video: Path, out_jsonl: Path) -> None:
    """Execute one job. Jobs are serialised because a single GPU cannot
    run several attention pipelines concurrently without thrashing."""
    with _job_slots:
        _processor.run(job, input_path, out_video, out_jsonl)


class _Slot:
    """A simple one-slot semaphore so jobs run one at a time."""

    def __init__(self) -> None:
        self._sem = threading.Semaphore(1)

    def __enter__(self):
        self._sem.acquire()
        return self

    def __exit__(self, *exc):
        self._sem.release()


_job_slots = _Slot()


@app.get("/")
def index():
    return render_template_string(PAGE_TEMPLATE)


@app.post("/api/upload")
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file part named 'video' in the request"}), 400

    file = request.files["video"]
    original_name = file.filename or "upload.mp4"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return (
            jsonify(
                {
                    "error": f"Unsupported file type '{suffix}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}"
                }
            ),
            400,
        )

    job_id = uuid.uuid4().hex[:12]
    stem = Path(original_name).stem[:60]
    input_path = OUTPUT_DIR / f"{job_id}_{stem}{suffix}"
    out_video = OUTPUT_DIR / f"{job_id}_{stem}_annotated.mp4"
    out_jsonl = OUTPUT_DIR / f"{job_id}_{stem}_scores.jsonl"

    file.save(input_path)
    size = input_path.stat().st_size
    if size == 0:
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "The uploaded file was empty."}), 400

    job = JobResult(
        job_id=job_id,
        status="queued",
        stage="Queued",
        message="Waiting for the analysis worker",
        input_name=original_name,
        input_path=str(input_path),
        queued_at=time.time(),
    )
    with _jobs_lock:
        _jobs[job_id] = job

    threading.Thread(
        target=_run_job,
        args=(job, input_path, out_video, out_jsonl),
        daemon=True,
    ).start()

    return jsonify({"job_id": job_id, "status": job.status, "input_name": original_name})


@app.get("/api/job/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job id"}), 404
    return jsonify(job.to_dict())


@app.get("/api/jobs")
def job_list():
    with _jobs_lock:
        jobs = sorted(_jobs.values(), key=lambda j: j.queued_at, reverse=True)
        return jsonify([j.to_dict() for j in jobs[:20]])


@app.get("/outputs/<path:name>")
def outputs(name: str):
    target = (OUTPUT_DIR / name).resolve()
    if not str(target).startswith(str(OUTPUT_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    if not target.exists():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(OUTPUT_DIR, name, conditional=True)


@app.get("/dashboard")
@app.get("/dashboard/")
def dashboard_root():
    if not (DASHBOARD_DIR / "index.html").exists():
        return jsonify({"error": "Dashboard not built yet"}), 404
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.get("/dashboard/<path:name>")
def dashboard_files(name: str):
    target = (DASHBOARD_DIR / name).resolve()
    if not str(target).startswith(str(DASHBOARD_DIR.resolve())):
        return jsonify({"error": "Invalid path"}), 400
    if not target.exists():
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(DASHBOARD_DIR, name)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File exceeds the 2 GB upload limit."}), 413


PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Attention Analysis — Upload</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
<style>
:root{
  color-scheme:dark;
  --bg:#070A12; --surface:rgba(19,25,40,.72); --surface-2:rgba(28,36,56,.66); --solid:#121827;
  --stroke:rgba(148,163,184,.12); --stroke-2:rgba(148,163,184,.22);
  --hi:#EEF2F8; --mid:#A7B3C7; --low:#6B7A93;
  --brand:#6D8BFF; --brand-2:#22D3EE; --brand-soft:rgba(109,139,255,.14);
  --ok:#34D399; --ok-soft:rgba(52,211,153,.13);
  --warn:#FBBF24; --warn-soft:rgba(251,191,36,.13);
  --bad:#FB7185; --bad-soft:rgba(251,113,133,.13);
  --r-md:12px; --r-lg:18px; --r-xl:26px;
  --ease:cubic-bezier(.22,1,.36,1);
  --font:'Plus Jakarta Sans',system-ui,-apple-system,sans-serif;
  --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{
  min-height:100vh;background:var(--bg);
  background-image:
    radial-gradient(1100px 620px at 8% -12%,rgba(109,139,255,.16),transparent 62%),
    radial-gradient(900px 520px at 96% 4%,rgba(34,211,238,.11),transparent 60%);
  background-attachment:fixed;color:var(--hi);font-family:var(--font);
  line-height:1.55;-webkit-font-smoothing:antialiased;
}
h1,h2,h3{line-height:1.2;letter-spacing:-.022em;font-weight:700}
:focus-visible{outline:2px solid var(--brand);outline-offset:2px;border-radius:6px}
.shell{max-width:1180px;margin:0 auto;padding:0 24px 72px}

/* header */
.masthead{
  position:sticky;top:0;z-index:40;display:flex;align-items:center;justify-content:space-between;
  gap:18px;padding:14px 24px;background:rgba(7,10,18,.72);
  backdrop-filter:blur(22px) saturate(160%);border-bottom:1px solid var(--stroke);
}
.brand{display:flex;align-items:center;gap:12px}
.mark{width:36px;height:36px;display:grid;place-items:center;border-radius:11px;
  background:linear-gradient(145deg,rgba(109,139,255,.28),rgba(34,211,238,.16));
  border:1px solid rgba(109,139,255,.36);color:var(--brand)}
.mark svg{width:18px;height:18px}
.brand b{font-size:.94rem;font-weight:800}
.brand small{display:block;font-size:.68rem;color:var(--low);text-transform:uppercase;letter-spacing:.1em;font-weight:600}
.badge{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;
  font-size:.66rem;font-weight:700;letter-spacing:.045em;text-transform:uppercase;
  background:var(--surface-2);border:1px solid var(--stroke);color:var(--mid)}
.badge.ok{background:var(--ok-soft);color:var(--ok);border-color:rgba(52,211,153,.28)}
.badge.run{background:var(--brand-soft);color:#9FB3FF;border-color:rgba(109,139,255,.3)}
.badge.bad{background:var(--bad-soft);color:var(--bad);border-color:rgba(251,113,133,.28)}

/* panel */
.panel{background:var(--surface);backdrop-filter:blur(18px) saturate(140%);
  border:1px solid var(--stroke);border-radius:var(--r-lg);
  box-shadow:0 1px 2px rgba(0,0,0,.34),0 12px 32px -12px rgba(0,0,0,.6);
  padding:22px 24px 24px;display:flex;flex-direction:column;gap:16px}
.panel h2{font-size:.98rem;display:flex;align-items:center;gap:9px}
.panel h2 .ico{width:29px;height:29px;display:grid;place-items:center;border-radius:9px;
  background:var(--surface-2);border:1px solid var(--stroke);color:var(--brand)}
.panel h2 .ico svg{width:15px;height:15px}
.sub{font-size:.77rem;color:var(--low);margin-top:3px}

/* dropzone */
.drop{
  position:relative;border:1.5px dashed var(--stroke-2);border-radius:var(--r-md);
  background:rgba(148,163,184,.03);padding:40px 24px;text-align:center;
  cursor:pointer;transition:border-color .2s var(--ease),background .2s var(--ease);
}
.drop:hover,.drop.over{border-color:rgba(109,139,255,.6);background:rgba(109,139,255,.06)}
.drop input{position:absolute;inset:0;opacity:0;cursor:pointer}
.drop .big{font-size:.86rem;font-weight:700;color:var(--hi);margin-top:12px}
.drop .hint{font-size:.75rem;color:var(--low);margin-top:4px}
.drop svg{width:34px;height:34px;color:var(--brand)}
.picked{margin-top:14px;display:none;align-items:center;justify-content:space-between;gap:14px;
  background:var(--surface-2);border:1px solid var(--stroke);border-radius:var(--r-md);padding:12px 15px}
.picked.on{display:flex}
.picked .nm{font-size:.83rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.picked .sz{font-size:.72rem;color:var(--low);font-family:var(--mono)}

/* button */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;
  background:linear-gradient(135deg,var(--brand),#5B7BFF);color:#fff;border:0;border-radius:10px;
  padding:12px 22px;font-size:.85rem;font-weight:700;font-family:var(--font);cursor:pointer;
  transition:transform .18s var(--ease),box-shadow .18s var(--ease),opacity .18s}
.btn:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 10px 26px -10px rgba(109,139,255,.85)}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn.ghost{background:var(--surface-2);border:1px solid var(--stroke);color:var(--mid);box-shadow:none}
.btn.ghost:hover:not(:disabled){color:var(--hi);border-color:var(--stroke-2)}
.btn svg{width:16px;height:16px}

/* progress */
.prog{display:none;flex-direction:column;gap:10px}
.prog.on{display:flex}
.track{height:7px;border-radius:99px;background:rgba(148,163,184,.14);overflow:hidden}
.fill{height:100%;width:0;border-radius:99px;
  background:linear-gradient(90deg,var(--brand),var(--brand-2));transition:width .35s var(--ease)}
.prog-row{display:flex;justify-content:space-between;gap:14px;font-size:.78rem;color:var(--mid)}
.prog-row b{font-family:var(--mono);color:var(--hi)}
.spin{width:13px;height:13px;border-radius:50%;border:2px solid rgba(109,139,255,.3);
  border-top-color:var(--brand);animation:sp .8s linear infinite;display:inline-block;vertical-align:-2px}
@keyframes sp{to{transform:rotate(360deg)}}

/* timing cards */
.timings{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.timing{background:var(--surface-2);border:1px solid var(--stroke);border-radius:var(--r-md);padding:14px 16px}
.timing .k{font-size:.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--low)}
.timing .v{font-family:var(--mono);font-size:1.32rem;font-weight:700;margin-top:5px;letter-spacing:-.02em}
.timing .n{font-size:.7rem;color:var(--low);margin-top:2px}

/* stats */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px}
.stat{background:var(--surface-2);border:1px solid var(--stroke);border-radius:var(--r-md);padding:13px 15px}
.stat .k{font-size:.61rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--low)}
.stat .v{font-family:var(--mono);font-size:1.24rem;font-weight:700;margin-top:4px;letter-spacing:-.02em}

/* video */
.video-wrap{position:relative;border-radius:var(--r-md);overflow:hidden;border:1px solid var(--stroke);background:#05070C}
video{width:100%;display:block;max-height:70vh;background:#05070C}
.vid-btns{display:flex;gap:10px;flex-wrap:wrap}

/* table */
table{width:100%;border-collapse:collapse;font-size:.83rem}
th{text-align:left;padding:10px 13px;color:var(--low);font-size:.66rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.07em;border-bottom:1px solid var(--stroke)}
td{padding:10px 13px;border-bottom:1px solid rgba(148,163,184,.07);color:var(--mid)}
tbody tr:last-child td{border-bottom:0}
.pill{display:inline-block;padding:3px 9px;border-radius:6px;font-size:.64rem;font-weight:700;
  letter-spacing:.05em;text-transform:uppercase;font-family:var(--mono);border:1px solid transparent}
.pill.ok{background:var(--ok-soft);color:var(--ok);border-color:rgba(52,211,153,.28)}
.pill.bad{background:var(--bad-soft);color:var(--bad);border-color:rgba(251,113,133,.28)}
.pill.warn{background:var(--warn-soft);color:var(--warn);border-color:rgba(251,191,36,.28)}

/* alert */
.err{display:none;background:var(--bad-soft);border:1px solid rgba(251,113,133,.35);
  border-radius:var(--r-md);padding:13px 16px;font-size:.82rem;color:#FECDD3}
.err.on{display:block}

/* section */
.sec-h{display:flex;align-items:baseline;justify-content:space-between;gap:16px;margin:30px 0 14px;flex-wrap:wrap}
.sec-h h2{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.13em;color:var(--low)}
.rule{flex:1;height:1px;background:var(--stroke);min-width:20px}
.hidden{display:none!important}
.chart{width:100%;height:210px}
.chart canvas{width:100%!important;height:100%!important;display:block}
@media(max-width:700px){.shell{padding:0 14px 50px}.masthead{padding:12px 14px}.panel{padding:18px 16px 20px}}
@media(prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style>
</head>
<body>

<header class="masthead">
  <div class="brand">
    <div class="mark">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 0-4 4v1H7a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-1V6a4 4 0 0 0-4-4Z"/><path d="M9 13h6"/><path d="M12 11v4"/></svg>
    </div>
    <div>
      <b>NEUROCLASS — Video Analysis</b>
      <small>Multimodal classroom attention engine</small>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <a class="btn ghost" href="/dashboard" style="padding:6px 13px;font-size:.75rem;text-decoration:none">Analytics Dashboard</a>
    <span class="badge" id="statusBadge">Idle</span>
  </div>
</header>

<main class="shell">

  <section class="panel" style="margin-top:24px">
    <div>
      <h2><span class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg></span>Upload a classroom video</h2>
      <div class="sub">The video is analysed by the six-stage attention pipeline and returned as an annotated, browser-playable MP4.</div>
    </div>

    <div class="drop" id="drop">
      <input type="file" id="file" accept="video/*,.mp4,.avi,.mov,.mkv,.webm" />
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="m16 11 6-3v8l-6-3z"/></svg>
      <div class="big">Drop a video here, or click to browse</div>
      <div class="hint">MP4, AVI, MOV, MKV or WEBM &middot; up to 2 GB</div>
    </div>

    <div class="picked" id="picked">
      <div style="min-width:0">
        <div class="nm" id="pickedName">—</div>
        <div class="sz" id="pickedSize">—</div>
        <div class="sz" id="pickedEta" style="margin-top:3px;color:var(--brand-2)"></div>
      </div>
      <button class="btn ghost" id="clearBtn" type="button">Change</button>
    </div>

    <div>
      <button class="btn" id="uploadBtn" type="button" disabled>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3v4"/><path d="M3 5h4"/><path d="M6 17v4"/><path d="M4 19h4"/><path d="M13 3v4"/><path d="M11 5h4"/><path d="M12 20v2"/><path d="M10 22h4"/><path d="m17 16 3 3 3-3"/><path d="M20 19v-6"/><path d="m14 15 3-3 3 3"/><path d="M17 12V6"/></svg>
        Analyse video
      </button>
    </div>

    <div class="prog" id="prog">
      <div class="track"><div class="fill" id="fill"></div></div>
      <div class="prog-row">
        <span id="stage"><span class="spin"></span> Preparing…</span>
        <b id="pct">0%</b>
      </div>
      <div class="prog-row" style="color:var(--low)">
        <span id="message">—</span>
        <span id="elapsed"></span>
      </div>
      <div class="prog-row" id="etaRow" style="display:none;color:var(--brand-2)">
        <span id="etaLabel">—</span>
        <span id="etaClock" class="mono">—</span>
      </div>
    </div>

    <div class="err" id="err"></div>
  </section>

  <div id="results" class="hidden">

    <div class="sec-h"><h2>Processing time</h2><div class="rule"></div></div>
    <section class="panel">
      <div class="timings" id="timings"></div>
      <div class="sub" id="timingNote" style="margin-top:2px"></div>
    </section>

    <div class="sec-h"><h2>Annotated output</h2><div class="rule"></div></div>
    <section class="panel">
      <div class="video-wrap">
        <video id="outVideo" controls playsinline preload="metadata"></video>
      </div>
      <div class="vid-btns">
        <a class="btn" id="downloadBtn" download>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/></svg>
          Download annotated video
        </a>
        <a class="btn ghost" id="scoresBtn" download>Download scores (JSONL)</a>
      </div>
      <div class="sub" id="videoNote"></div>
    </section>

    <div class="sec-h"><h2>Attention summary</h2><div class="rule"></div></div>
    <section class="panel">
      <div class="stats" id="stats"></div>
      <div>
        <div class="sub" style="margin-bottom:8px">Attention score over time</div>
        <div class="chart"><canvas id="chart"></canvas></div>
      </div>
    </section>

    <div class="sec-h"><h2>Attention-drop events</h2><div class="rule"></div></div>
    <section class="panel">
      <div class="table-wrap" style="overflow-x:auto">
        <table>
          <thead><tr><th>Time</th><th>Student</th><th>Kind</th><th>Score</th><th>Pedagogical move</th></tr></thead>
          <tbody id="eventsBody"></tbody>
        </table>
      </div>
    </section>

    <div class="sec-h"><h2>Student registry</h2><div class="rule"></div></div>
    <section class="panel">
      <div class="table-wrap" style="overflow-x:auto">
        <table>
          <thead><tr><th>Student</th><th>Mean score</th><th>Attentive %</th><th>Min / max</th><th>Dominant behaviour</th></tr></thead>
          <tbody id="studentsBody"></tbody>
        </table>
      </div>
    </section>

  </div>

  <footer class="sub" style="margin-top:38px;padding-top:18px;border-top:1px solid var(--stroke)">
    Attention is an observable behavioural surrogate, not a measure of internal cognitive state. Use for pedagogical reflection only.
  </footer>
</main>

<script>
const $ = (id) => document.getElementById(id);
let selectedFile = null;
let pollTimer = null;
let startedAt = 0;
let etaSeconds = null;   // remaining seconds, refreshed from the server
let etaTick = null;

// Measured throughput of this pipeline on this GPU (video seconds per
// wall-clock second). Used only to give the user a wait estimate.
const REALTIME_FACTOR = 0.44;

const fmtBytes = (b) => {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  if (b < 1073741824) return (b / 1048576).toFixed(1) + ' MB';
  return (b / 1073741824).toFixed(2) + ' GB';
};
const fmtTime = (s) => {
  if (s === null || s === undefined) return '—';
  if (s < 0) s = 0;
  if (s < 60) return s.toFixed(1) + 's';
  const m = Math.floor(s / 60);
  return m + 'm ' + Math.round(s - m * 60) + 's';
};

/* ---- file selection ---- */
const drop = $('drop');
const fileInput = $('file');

drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) setFile(fileInput.files[0]); });

function setFile(f) {
  selectedFile = f;
  $('pickedName').textContent = f.name;
  $('pickedSize').textContent = fmtBytes(f.size);
  $('picked').classList.add('on');
  $('uploadBtn').disabled = false;
  estimateBeforeUpload(f);
}

/* Probe the local file's duration so the wait estimate is real rather than
   guessed from file size. */
function estimateBeforeUpload(f) {
  const url = URL.createObjectURL(f);
  const v = document.createElement('video');
  v.preload = 'metadata';
  v.onloadedmetadata = () => {
    URL.revokeObjectURL(url);
    const dur = isFinite(v.duration) ? v.duration : null;
    if (!dur) { $('pickedEta').textContent = ''; return; }
    const wait = dur / REALTIME_FACTOR;
    $('pickedEta').textContent =
      dur.toFixed(1) + 's of video · about ' + fmtTime(wait) + ' to analyse';
  };
  v.onerror = () => {
    URL.revokeObjectURL(url);
    $('pickedEta').textContent = '';
  };
  v.src = url;
}
$('clearBtn').addEventListener('click', () => {
  selectedFile = null; fileInput.value = '';
  $('picked').classList.remove('on');
  $('uploadBtn').disabled = true;
});

/* ---- upload ---- */
$('uploadBtn').addEventListener('click', async () => {
  if (!selectedFile) return;
  $('uploadBtn').disabled = true;
  $('err').classList.remove('on');
  $('results').classList.add('hidden');
  $('prog').classList.add('on');
  $('fill').style.width = '0%';
  $('pct').textContent = '0%';
  $('stage').innerHTML = '<span class="spin"></span> Uploading…';
  $('message').textContent = 'Transferring file to the analysis server';
  startedAt = Date.now();
  setBadge('run', 'Uploading');
  $('etaRow').style.display = 'none';
  etaSeconds = null;
  clearInterval(etaTick);
  etaTick = setInterval(() => {
    if (etaSeconds === null) return;
    etaSeconds = Math.max(0, etaSeconds - 1);
    $('etaClock').textContent = fmtTime(etaSeconds);
  }, 1000);

  const fd = new FormData();
  fd.append('video', selectedFile);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('Upload failed (' + res.status + ')'));
    poll(data.job_id);
  } catch (e) {
    fail(e.message);
  }
});

/* ---- polling ---- */
function poll(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/job/' + jobId);
      if (!res.ok) throw new Error('Could not read job status');
      const j = await res.json();
      render(j);
      if (j.status === 'done' || j.status === 'error') {
        clearInterval(pollTimer);
        if (j.status === 'error') fail(j.error || 'Processing failed');
        else finish(j);
      }
    } catch (e) {
      clearInterval(pollTimer);
      fail(e.message);
    }
  }, 700);
}

function render(j) {
  const pct = Math.round((j.progress || 0) * 100);
  $('fill').style.width = pct + '%';
  $('pct').textContent = pct + '%';
  $('stage').innerHTML = '<span class="spin"></span> ' + (j.stage || 'Processing');
  $('message').textContent = j.message || '';
  const el = (Date.now() - startedAt) / 1000;
  $('elapsed').textContent = 'elapsed ' + fmtTime(el);

  // Live remaining-time estimate, derived from the server's own progress
  // and the measured throughput of this pipeline.
  const v = j.video || {};
  const t = j.timing || {};
  if (j.status === 'running' && v.duration_s && (j.progress || 0) > 0.05) {
    const done = j.progress;
    const remaining = (1 - done) * (t.total_s || el);
    etaSeconds = Math.max(0, remaining);
    $('etaRow').style.display = 'flex';
    $('etaLabel').textContent = 'Estimated remaining';
    $('etaClock').textContent = fmtTime(etaSeconds);
  } else if (j.status === 'running' && v.duration_s) {
    etaSeconds = v.duration_s / REALTIME_FACTOR;
    $('etaRow').style.display = 'flex';
    $('etaLabel').textContent = 'Estimated total time';
    $('etaClock').textContent = fmtTime(etaSeconds);
  }

  setBadge('run', j.status === 'queued' ? 'Queued' : 'Processing');
}

function finish(j) {
  clearInterval(pollTimer);
  clearInterval(etaTick);
  $('etaRow').style.display = 'none';
  setBadge('ok', 'Complete');
  $('prog').classList.remove('on');
  $('results').classList.remove('hidden');

  const t = j.timing, v = j.video, r = j.results;

  $('timings').innerHTML = [
    ['Total time taken', fmtTime(t.total_s), 'upload → analysis → encode'],
    ['Model load', fmtTime(t.model_load_s), 'weights &amp; CUDA init'],
    ['Analysis', fmtTime(t.inference_s), '6-stage pipeline'],
    ['Video encoding', fmtTime(t.encode_s), 'H.264 mux for the browser'],
  ].map(([k, val, n]) =>
    '<div class="timing"><div class="k">' + k + '</div><div class="v">' + val + '</div><div class="n">' + n + '</div></div>'
  ).join('');

  const rt = v.realtime_factor || 0;
  const speed = v.processing_fps || 0;
  const vidSecs = v.duration_s || 0;
  const perMin = rt > 0 ? (60 / rt) : 0;
  $('timingNote').innerHTML =
    'Analysed <b>' + vidSecs.toFixed(1) + 's</b> of video at <b>' + speed.toFixed(1) +
    ' frames/second</b> — about <b>' + rt.toFixed(2) + 'x realtime</b> on this GPU' +
    (rt < 1 ? ', so roughly <b>' + perMin.toFixed(0) + 's of waiting per minute of video</b>.' : '.') +
    ' The pipeline runs the full six stages (detection &amp; tracking, gaze, posture, ' +
    'behaviour, multimodal scoring, drop detection) plus H.264 encoding.';

  $('outVideo').src = '/outputs/' + j.output_video;
  $('downloadBtn').href = '/outputs/' + j.output_video;
  $('downloadBtn').setAttribute('download', j.input_name.replace(/\.[^.]+$/, '') + '_annotated.mp4');
  $('scoresBtn').href = '/outputs/' + j.output_jsonl;
  $('scoresBtn').setAttribute('download', j.input_name.replace(/\.[^.]+$/, '') + '_scores.jsonl');
  $('videoNote').textContent =
    'Source: ' + j.input_name + ' · ' + v.width + '×' + v.height + ' @ ' + v.fps +
    ' fps · ' + (v.frames_processed || 0).toLocaleString() + ' frames annotated.';

  $('stats').innerHTML = [
    ['Mean attention', (r.mean_score || 0).toFixed(1) + '/100'],
    ['Attentive', (r.attentive_pct || 0).toFixed(1) + '%'],
    ['Students tracked', r.students_tracked || 0],
    ['Observations', (r.observations || 0).toLocaleString()],
    ['Drop events', (r.drop_events || []).length],
  ].map(([k, val]) => '<div class="stat"><div class="k">' + k + '</div><div class="v">' + val + '</div></div>').join('');

  const eb = $('eventsBody');
  eb.innerHTML = '';
  if (!(r.drop_events || []).length) {
    eb.innerHTML = '<tr><td colspan="5" style="color:var(--low);padding:18px 13px">No attention-drop events were detected in this clip.</td></tr>';
  } else {
    r.drop_events.forEach(e => {
      const tr = document.createElement('tr');
      const kindClass = e.kind === 'sudden' ? 'bad' : 'warn';
      tr.innerHTML =
        '<td class="mono">' + e.timestamp.toFixed(1) + 's</td>' +
        '<td class="mono">S' + String(e.student_id).padStart(2, '0') + '</td>' +
        '<td><span class="pill ' + kindClass + '">' + e.kind + '</span></td>' +
        '<td class="mono">' + e.score.toFixed(1) + '</td>' +
        '<td>' + (e.recommendation || '—') + '</td>';
      eb.appendChild(tr);
    });
  }

  const sb = $('studentsBody');
  sb.innerHTML = '';
  if (!(r.students || []).length) {
    sb.innerHTML = '<tr><td colspan="5" style="color:var(--low);padding:18px 13px">No students were tracked in this clip.</td></tr>';
  } else {
    r.students.forEach(s => {
      const high = s.mean_score >= 50;
      const tr = document.createElement('tr');
      tr.innerHTML =
        '<td class="mono">S' + String(s.id).padStart(2, '0') + '</td>' +
        '<td class="mono" style="color:' + (high ? 'var(--ok)' : 'var(--bad)') + '">' + s.mean_score.toFixed(1) + '</td>' +
        '<td class="mono">' + s.attentive_pct.toFixed(1) + '%</td>' +
        '<td class="mono" style="color:var(--low)">' + s.min_score.toFixed(1) + ' / ' + s.max_score.toFixed(1) + '</td>' +
        '<td>' + (s.behavior || s.cue || '—') + '</td>';
      sb.appendChild(tr);
    });
  }

  drawChart(r.timeline || []);
  $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function drawChart(timeline) {
  const canvas = $('chart');
  const ctx = canvas.getContext('2d');
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (timeline.length < 2) return;

  const W = rect.width, H = rect.height;
  const pad = { t: 14, r: 14, b: 28, l: 34 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const t0 = timeline[0].t, t1 = timeline[timeline.length - 1].t;
  const span = Math.max(0.001, t1 - t0);
  const X = t => pad.l + ((t - t0) / span) * pw;
  const Y = v => pad.t + ph - (Math.max(0, Math.min(100, v)) / 100) * ph;

  ctx.font = '10px "JetBrains Mono", monospace';
  ctx.textBaseline = 'middle';
  for (let v = 0; v <= 100; v += 25) {
    const y = Y(v);
    ctx.strokeStyle = v === 50 ? 'rgba(148,163,184,.18)' : 'rgba(148,163,184,.07)';
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.fillStyle = '#6B7A93'; ctx.textAlign = 'right';
    ctx.fillText(String(v), pad.l - 7, y);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const ticks = Math.min(6, Math.max(2, Math.floor(pw / 90)));
  for (let i = 0; i <= ticks; i++) {
    const t = t0 + span * i / ticks;
    ctx.fillStyle = '#6B7A93';
    ctx.fillText(t.toFixed(1) + 's', X(t), H - pad.b + 8);
  }

  const pts = timeline.map(p => ({ x: X(p.t), y: Y(p.mean_score) }));

  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ph);
  grad.addColorStop(0, 'rgba(109,139,255,.34)');
  grad.addColorStop(1, 'rgba(34,211,238,0)');
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pad.t + ph);
  pts.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(pts[pts.length - 1].x, pad.t + ph);
  ctx.closePath(); ctx.fillStyle = grad; ctx.fill();

  ctx.beginPath();
  ctx.strokeStyle = '#8AA4FF'; ctx.lineWidth = 2.2; ctx.lineJoin = 'round';
  pts.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y)));
  ctx.stroke();
}

function fail(msg) {
  clearInterval(pollTimer);
  clearInterval(etaTick);
  $('etaRow').style.display = 'none';
  $('prog').classList.remove('on');
  $('err').textContent = msg;
  $('err').classList.add('on');
  $('uploadBtn').disabled = false;
  setBadge('bad', 'Failed');
}

function setBadge(kind, text) {
  const b = $('statusBadge');
  b.className = 'badge ' + kind;
  b.textContent = text;
}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Video upload attention analysis server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(" VIDEO ANALYSIS SERVER")
    print("=" * 70)
    print(f" Open: http://{args.host}:{args.port}")
    print(f" Outputs: {OUTPUT_DIR}")
    print(f" Uploads are purged after {RETENTION_HOURS} hours")
    print("=" * 70)

    _purge_old_outputs()
    threading.Thread(target=_purge_loop, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
