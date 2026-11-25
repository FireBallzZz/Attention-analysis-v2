"""Launches the video-upload attention analysis server and opens the browser.

    python launch_video_upload.py

Options:
    --port N   port to listen on (default 8000)
    --host H   interface to bind (default 127.0.0.1)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
SERVER = WORKSPACE_ROOT / "scripts" / "web_upload_server.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the video upload analysis server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    def open_browser() -> None:
        # Give the server a moment to bind before opening the page.
        time.sleep(2.0)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if not args.no_browser:
        threading.Thread(target=open_browser, daemon=True).start()

    print("=" * 70)
    print(" VIDEO ANALYSIS SERVER")
    print("=" * 70)
    print(f" Open: {url}")
    print(" Upload a classroom video to get an annotated MP4 and timing report.")
    print(" Press Ctrl+C to stop.")
    print("=" * 70)

    subprocess.run(
        [sys.executable, str(SERVER), "--host", args.host, "--port", str(args.port)],
        cwd=str(SERVER.parent),
    )


if __name__ == "__main__":
    main()
