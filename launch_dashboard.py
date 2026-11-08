"""Launches the NEUROCLASS AI Multimodal Attention Intelligence Dashboard in default web browser."""

import webbrowser
from pathlib import Path

def main():
    dashboard_path = Path(__file__).resolve().parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        print(f"Dashboard not found at: {dashboard_path}")
        return
    print(f"Launching NEUROCLASS AI Dashboard: {dashboard_path.resolve()}")
    webbrowser.open(dashboard_path.as_uri())

if __name__ == "__main__":
    main()
