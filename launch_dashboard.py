"""Launches the NEUROCLASS AI Multimodal Attention Intelligence Dashboard.

Regenerates the dataset and HTML from the real run artifacts first, so the
dashboard always reflects the latest evaluation results.
"""

import subprocess
import sys
import webbrowser
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
DASHBOARD_PATH = WORKSPACE_ROOT / "dashboard" / "index.html"
BUILDER = WORKSPACE_ROOT / "scripts" / "build_dashboard_dataset.py"
GENERATOR = WORKSPACE_ROOT / "scripts" / "generate_dashboard_html.py"


def run_step(label: str, script: Path) -> bool:
    """Run a build step; warn but continue if it fails."""
    print(f"[{label}] {script.name}")
    result = subprocess.run(
        [sys.executable, str(script)], cwd=str(script.parent)
    )
    if result.returncode != 0:
        print(f"  warning: {script.name} exited with {result.returncode}")
        return False
    return True


def main():
    run_step("dataset", BUILDER)
    run_step("html", GENERATOR)

    if not DASHBOARD_PATH.exists():
        print(f"Dashboard not found at: {DASHBOARD_PATH}")
        return

    print(f"Launching NEUROCLASS AI Dashboard: {DASHBOARD_PATH.resolve()}")
    webbrowser.open(DASHBOARD_PATH.as_uri())


if __name__ == "__main__":
    main()
