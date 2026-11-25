"""Download and merge the SCB classroom-behaviour datasets into one trainable set.

Why this exists
---------------
The deployed checkpoint ``checkpoints/scb_yolo.pt`` was trained on the 3-class
SCB subset (``hand-raising``, ``read``, ``write``). The pipeline's feature
taxonomy has six posture channels, so ``using-phone``, ``bowing-head`` and
``leaning-over-table`` could never be produced by the detector - the pipeline
fell back to geometric proxies for all three.

Upstream (https://huggingface.co/datasets/wintonYF/SCB-Dataset) ships the
missing behaviour as a separate detector:

  * ``SCB5-Handrise-Read-write-2024-9-17`` -> hand-raising, reading, writing
  * ``SCB_BowTurnHead_20250509``            -> BowHead, TurnHead

This script downloads both, merges them into a single 4-class dataset
(hand-raising, reading, writing, bowing-head) that fits the existing six-slot
posture vector, and writes the Ultralytics YAML.

Usage:
    python scripts/setup_scb_dataset.py
    python scripts/setup_scb_dataset.py --skip-download   # merge only
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "scb"

HF_REPO = "wintonYF/SCB-Dataset"

HANDWRITE_ZIP = "SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip"
BOWTURN_ZIP = "SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509.zip"

# The merged taxonomy. Index = class id used in the merged label files.
# These are a subset of attention_pipeline.types.SCB_BEHAVIOR_LABELS, so the
# merged detector's outputs map straight onto the existing posture channels.
MERGED_NAMES = ["hand-raising", "reading", "writing", "bowing-head"]

# Source dataset -> (label subdir, {source class id: merged class id})
HANDWRITE_ROOT = "SCB5-Handrise-Read-write-2024-9-17"
BOWTURN_ROOT = "SCB_BowTurnHead_20250509/SCB5-Turn-Bow-Head-2024-9-17"

HANDWRITE_MAP = {0: 0, 1: 1, 2: 2}          # hand-raising, read, write
BOWTURN_MAP = {0: 3}                         # BowHead -> bowing-head


def _download(repo_id: str, filename: str, dest: Path) -> Path:
    """Fetch one file from the Hugging Face hub, skipping if already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"Already have {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"Downloading {repo_id}/{filename}")
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        shutil.copy2(cached, dest)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  huggingface_hub unavailable ({exc}); falling back to a direct URL")
        from urllib.request import urlretrieve

        url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
        urlretrieve(url, dest)
    print(f"  saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _extract(zip_path: Path, out_dir: Path) -> Path:
    """Extract an archive once and return the directory holding images/labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "EXTRACTED"
    if not marker.exists():
        print(f"Extracting {zip_path.name} -> {out_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        marker.write_text(zip_path.name, encoding="utf-8")
    else:
        print(f"Already extracted: {out_dir}")

    for candidate in out_dir.rglob("images"):
        if (candidate / "train").exists():
            return candidate.parent
    raise FileNotFoundError(f"No images/train found under {out_dir}")


def _remap_labels(src_labels: Path, mapping: dict[int, int]) -> list[str]:
    """Rewrite one label file's class ids into the merged taxonomy."""
    out_rows: list[str] = []
    for line in src_labels.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            src_id = int(float(parts[0]))
        except ValueError:
            continue
        new_id = mapping.get(src_id)
        if new_id is None:
            continue
        out_rows.append(" ".join([str(new_id)] + parts[1:]))
    return out_rows


def _merge_split(
    sources: list[tuple[Path, dict[int, int], str]],
    split: str,
    out_root: Path,
) -> tuple[int, int]:
    """Merge one split of one or more sources into the unified dataset."""
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    n_skipped = 0
    for src_root, mapping, tag in sources:
        src_images = src_root / "images" / split
        src_labels = src_root / "labels" / split
        if not src_images.exists():
            print(f"  [skip] {tag}/{split}: no images")
            continue
        for img in sorted(src_images.glob("*")):
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            lbl = src_labels / f"{img.stem}.txt"
            if not lbl.exists():
                n_skipped += 1
                continue
            rows = _remap_labels(lbl, mapping)
            if not rows:
                # An image whose boxes all fall outside the merged taxonomy is
                # dropped: keeping it would teach the detector that visible
                # students are background.
                n_skipped += 1
                continue
            # Prefix with the source tag so the two archives cannot collide.
            stem = f"{tag}_{img.stem}"
            shutil.copy2(img, img_out / f"{stem}{img.suffix.lower()}")
            (lbl_out / f"{stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            n_boxes += len(rows)
            n_images += 1
    print(f"  {split}: skipped {n_skipped} images with no usable annotation")
    return n_images, n_boxes


def _write_yaml(out_root: Path) -> Path:
    path = out_root / "scb.yaml"
    path.write_text(
        f"path: {str(out_root).replace(chr(92), '/')}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(MERGED_NAMES)}\n"
        f"names: {MERGED_NAMES}\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--out", default=str(RAW / "scb_merged_4class"))
    args = parser.parse_args()

    out_root = Path(args.out)
    # Always rebuild the merged dataset so a re-run cannot leave stale images
    # behind from a previous merge that had a different filter.
    if out_root.exists():
        print(f"Removing previous merged dataset at {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        _download(HF_REPO, HANDWRITE_ZIP, RAW / "SCB5-Handrise-Read-write.zip")
        _download(HF_REPO, BOWTURN_ZIP, RAW / "SCB_BowTurnHead.zip")

    handwrite_zip = RAW / "SCB5-Handrise-Read-write.zip"
    bowturn_zip = RAW / "SCB_BowTurnHead.zip"
    if not handwrite_zip.exists():
        raise SystemExit(f"Missing {handwrite_zip}. Run without --skip-download.")
    if not bowturn_zip.exists():
        raise SystemExit(f"Missing {bowturn_zip}. Run without --skip-download.")

    handwrite_root = _extract(handwrite_zip, RAW / "SCB5-Handrise-Read-write")
    bowturn_root = _extract(bowturn_zip, RAW / "SCB_BowTurnHead")

    sources = [
        (handwrite_root, HANDWRITE_MAP, "hrw"),
        (bowturn_root, BOWTURN_MAP, "bow"),
    ]

    print()
    for split in ("train", "val"):
        n_images, n_boxes = _merge_split(sources, split, out_root)
        print(f"{split}: {n_images} images, {n_boxes} boxes")

    yaml_path = _write_yaml(out_root)
    print(f"\nMerged dataset: {out_root}")
    print(f"YAML          : {yaml_path}")
    print(f"Classes       : {MERGED_NAMES}")


if __name__ == "__main__":
    main()
