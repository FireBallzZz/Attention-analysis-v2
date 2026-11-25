"""Train inattention-cue detectors on SCB-Dataset5 (official Hugging Face release,
https://huggingface.co/datasets/wintonYF/SCB-Dataset):

  scb5_bowturn.pt : BowHead, TurnHead   (SCB_BowTurnHead_20250509, 2,410 images)
  scb5_discuss.pt : discuss             (SCB5-Discuss-2024-9-17,     864 images)

Each part is labelled only for its own classes, so the two detectors are trained
separately (merging would teach the model that unlabelled behaviours are
background). Starts from the local COCO-pretrained yolo11n.pt.

    python scripts/train_scb5_detectors.py
Writes checkpoints/scb5_bowturn.pt, checkpoints/scb5_discuss.pt and
reports/improvement/scb5_training.json (validation mAP).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "scb"
PARTS = {
    "bowturn": (RAW / "SCB_BowTurnHead" / "SCB_BowTurnHead_20250509" / "SCB5-Turn-Bow-Head-2024-9-17", ["BowHead", "TurnHead"]),
    "discuss": (RAW / "SCB5-Discuss" / "extracted" / "SCB5-Discuss-2024-9-17", ["discuss"]),
}


def main(epochs: int = 40) -> None:
    os.chdir(ROOT)  # yolo11n.pt (also used by the AMP check) lives here
    from ultralytics import YOLO

    out = {}
    for name, (base, classes) in PARTS.items():
        yml = ROOT / "data" / "raw" / "scb" / f"scb5_{name}.yaml"
        yml.write_text(
            f"path: {base.as_posix()}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(classes)}\nnames: {classes}\n", encoding="utf-8")
        model = YOLO(str(ROOT / "yolo11n.pt"))
        model.train(data=str(yml), epochs=epochs, imgsz=640, batch=16, workers=2, plots=False,
                    project=str(ROOT / "runs" / "scb5_train"), name=name, exist_ok=True, verbose=False)
        best = ROOT / "runs" / "scb5_train" / name / "weights" / "best.pt"
        shutil.copy(best, ROOT / "checkpoints" / f"scb5_{name}.pt")
        m = YOLO(str(best)).val(data=str(yml), imgsz=640, plots=False, verbose=False)
        out[name] = {"classes": classes, "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                     "per_class_mAP50": {c: round(float(v), 4) for c, v in zip(classes, m.box.ap50)}}
        print(name, out[name])
    (ROOT / "reports" / "improvement" / "scb5_training.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
