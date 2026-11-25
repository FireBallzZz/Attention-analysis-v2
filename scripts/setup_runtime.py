"""One-time setup after cloning: fetch the YOLOv7 code the SCB behaviour model needs.

The model weights are already in the repository. checkpoints/scb_yolo.pt is a
YOLOv7 checkpoint, so it needs the YOLOv7 source code in third_party/yolov7
(pinned to the exact commit this project was built and evaluated with).
No datasets are downloaded.

    python scripts/setup_runtime.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLOV7_GIT = "https://github.com/WongKinYiu/yolov7.git"
YOLOV7_COMMIT = "a207844b1ce82d204ab36d87d496728d3d2348e7"
REQUIRED = [
    "yolo11n-pose.pt",
    "yolo11n.pt",
    "checkpoints/motion_rule_v1.json",
    "checkpoints/final_human_v1.pt",
    "checkpoints/l2cs_finetuned.pt",
    "checkpoints/scb_yolo.pt",
    "checkpoints/baseline/fusion_daisee.pt",
]


def main() -> None:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        sys.exit("Missing model files (was the repository cloned completely?):\n  " + "\n  ".join(missing))
    dest = ROOT / "third_party" / "yolov7"
    if (dest / "models" / "yolo.py").exists():
        print(f"YOLOv7 already present: {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(["git", "clone", "--filter=blob:none", YOLOV7_GIT, str(dest)])
    subprocess.check_call(["git", "-C", str(dest), "checkout", "--quiet", YOLOV7_COMMIT])
    print(f"YOLOv7 at {YOLOV7_COMMIT[:7]}. Setup complete - all model files present.")
    print('Try:  python analyse_video.py "path/to/classroom.mp4"')


if __name__ == "__main__":
    main()
