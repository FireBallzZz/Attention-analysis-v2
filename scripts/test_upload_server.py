"""End-to-end tests for the video-upload analysis server.

These run against a live server instance. Start it first with:

    python scripts/web_upload_server.py --port 8000

Then run:

    python scripts/test_upload_server.py

The tests verify the upload contract, the timing report, and that the
annotated output is a browser-playable H.264 MP4.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8000"
SAMPLE_VIDEO = WORKSPACE_ROOT / "runs" / "Test 6.mp4"

failures: list[str] = []
notes: list[str] = []


def check(condition: bool, label: str) -> None:
    (notes if condition else failures).append(
        ("  [ok]   " if condition else "  [FAIL] ") + label
    )


def upload_and_wait(path: Path, timeout: float = 900.0) -> dict:
    with path.open("rb") as handle:
        response = requests.post(
            f"{BASE}/api/upload",
            files={"video": (path.name, handle, "video/mp4")},
            timeout=timeout,
        )
    if response.status_code != 200:
        raise RuntimeError(f"upload rejected: HTTP {response.status_code} {response.text[:300]}")

    job_id = response.json()["job_id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = requests.get(f"{BASE}/api/job/{job_id}", timeout=30).json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(1.0)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def main() -> int:
    if not SAMPLE_VIDEO.exists():
        print(f"Sample video missing: {SAMPLE_VIDEO}")
        return 1

    print("=" * 70)
    print("UPLOAD SERVER END-TO-END TEST")
    print("=" * 70)

    # --- reject a non-video upload -------------------------------------
    bad = requests.post(
        f"{BASE}/api/upload",
        files={"video": ("notes.txt", b"not a video", "text/plain")},
        timeout=60,
    )
    check(bad.status_code == 400, f"non-video upload rejected (HTTP {bad.status_code})")

    # --- happy path ----------------------------------------------------
    job = upload_and_wait(SAMPLE_VIDEO)
    check(job["status"] == "done", f"job completed (status={job['status']})")
    if job["status"] != "done":
        print(job.get("error"))
        return 1

    timing = job["timing"]
    video = job["video"]
    results = job["results"]

    print()
    print("TIMINGS")
    for key, value in timing.items():
        print(f"  {key:<14} {value}")
    print("VIDEO")
    for key, value in video.items():
        print(f"  {key:<16} {value}")
    print("RESULTS")
    print(f"  students_tracked  {results['students_tracked']}")
    print(f"  observations      {results['observations']}")
    print(f"  attentive_pct     {results['attentive_pct']}")
    print(f"  mean_score        {results['mean_score']}")
    print(f"  drop_events       {len(results['drop_events'])}")
    print()

    # --- timing contract ------------------------------------------------
    check(timing["total_s"] > 0, f"total time reported: {timing['total_s']}s")
    check(timing["model_load_s"] >= 0, f"model load time reported: {timing['model_load_s']}s")
    check(timing["inference_s"] > 0, f"analysis time reported: {timing['inference_s']}s")
    check(
        timing["total_s"] >= timing["inference_s"],
        "total time is at least the analysis time",
    )
    check(video["realtime_factor"] > 0, f"realtime factor: {video['realtime_factor']}x")
    check(
        abs(video["realtime_factor"] - video["duration_s"] / timing["inference_s"]) < 0.05,
        "realtime factor is consistent with duration / analysis time",
    )

    # --- results contract -----------------------------------------------
    check(results["students_tracked"] > 0, f"students tracked: {results['students_tracked']}")
    check(results["observations"] > 0, f"observations: {results['observations']}")
    check(len(results["timeline"]) > 1, f"timeline points: {len(results['timeline'])}")

    # --- output video ---------------------------------------------------
    response = requests.get(f"{BASE}/outputs/{job['output_video']}", timeout=180)
    check(response.status_code == 200, f"output video served (HTTP {response.status_code})")

    output_path = WORKSPACE_ROOT / "runs" / "web_uploads" / job["output_video"]
    check(output_path.exists(), f"output video on disk: {output_path.name}")
    if output_path.exists():
        head = output_path.read_bytes()[:4096]
        check(
            b"avc1" in head or b"avcC" in head,
            "output is H.264/avc1 so browsers can play it inline",
        )

        import cv2

        capture = cv2.VideoCapture(str(output_path))
        frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frames += 1
        capture.release()
        check(frames > 0, f"output decodes back cleanly: {frames} frames")

    # --- scores ---------------------------------------------------------
    scores = requests.get(f"{BASE}/outputs/{job['output_jsonl']}", timeout=120)
    check(scores.status_code == 200, f"scores JSONL served (HTTP {scores.status_code})")
    rows = [json.loads(line) for line in scores.text.splitlines() if line.strip()]
    check(len(rows) > 0, f"scores JSONL parsed: {len(rows)} rows")
    check("students" in rows[0], "scores JSONL row schema intact")

    # --- determinism: the same video must give the same answer ----------
    second = upload_and_wait(SAMPLE_VIDEO)
    check(second["status"] == "done", "second upload of the same video completed")
    if second["status"] == "done":
        check(
            second["results"]["students_tracked"] == results["students_tracked"],
            "same video gives the same student count "
            f"({second['results']['students_tracked']} vs {results['students_tracked']})",
        )
        check(
            second["results"]["mean_score"] == results["mean_score"],
            "same video gives the same mean score",
        )
        check(
            len(second["results"]["drop_events"]) == len(results["drop_events"]),
            "same video gives the same number of drop events",
        )
        check(
            second["timing"]["model_load_s"] == 0.0,
            "second upload reuses the already-loaded pipeline",
        )

    # --- report ---------------------------------------------------------
    print()
    print("=" * 70)
    for note in notes:
        print(note)
    if failures:
        print()
        for failure in failures:
            print(failure)
        print()
        print(f"RESULT: {len(failures)} failure(s)")
        return 1
    print()
    print("RESULT: all end-to-end checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
