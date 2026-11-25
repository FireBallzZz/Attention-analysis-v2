"""Train the merged SCB classroom-behaviour detector.

The deployed `checkpoints/scb_yolo.pt` only resolves three classes
(`hand-raising`, `reading`, `writing`), so the pipeline's posture channels
could never express `using-phone`, `bowing-head` or `leaning-over-table` and
the rubric fell back to geometric proxies for all three.

`scripts/setup_scb_dataset.py` merges the upstream SCB archives into a
four-class dataset (`hand-raising`, `reading`, `writing`, `bowing-head`). This
script trains a YOLO11 detector on that dataset.

The result is written to `checkpoints/scb_yolo_4class.pt` and is NOT promoted
over the deployed checkpoint automatically: the merge changes the class
vocabulary, so it has to be validated on the classroom footage first.

Usage:
    python scripts/train_scb_merged.py
    python scripts/train_scb_merged.py --epochs 40 --imgsz 960
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from attention_pipeline.timed_checkpoint import TimedCheckpoint, resume_path_for
from attention_pipeline.utils import PROJECT_ROOT

ROOT = PROJECT_ROOT
DEFAULT_DATA = ROOT / "data" / "raw" / "scb" / "scb_merged_4class" / "scb.yaml"
DEFAULT_OUT = ROOT / "checkpoints" / "scb_yolo_4class.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0,
                        help="Dataloader workers. 0 avoids the Windows worker crash.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--checkpoint-minutes", type=float, default=30)
    parser.add_argument("--fresh", action="store_true", help="Ignore a saved resume checkpoint")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(
            f"Missing {data}\nRun: python scripts/setup_scb_dataset.py"
        )

    from ultralytics import YOLO

    out = Path(args.out)
    run_last = ROOT / "runs" / "scb" / "merged_4class" / "weights" / "last.pt"
    resume_copy = resume_path_for(out)

    weights = args.model
    resume = False
    if not args.fresh:
        if run_last.exists():
            weights = str(run_last)
            resume = True
        elif resume_copy.exists():
            weights = str(resume_copy)
            resume = True
    if resume:
        print(f"Resuming from {weights}")
    else:
        print(f"Checkpoint every {args.checkpoint_minutes:g} min -> {resume_copy}")

    model = YOLO(weights)
    clock = TimedCheckpoint(resume_copy, minutes=args.checkpoint_minutes)

    def _save_half_hour(trainer) -> None:
        if not clock.due():
            return
        trainer.save_model()
        clock.mark()
        last = Path(trainer.last)
        if not last.exists():
            return
        resume_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(last, resume_copy)
        print(f"Checkpoint saved -> {resume_copy}")

    model.add_callback("on_train_batch_end", _save_half_hour)
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=str(ROOT / "runs" / "scb"),
        name="merged_4class",
        exist_ok=True,
        resume=resume,
        patience=20,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out)
        print(f"Copied {best} -> {out}")
    else:
        print(f"Training finished; copy {best} to {out} manually")


if __name__ == "__main__":
    main()
