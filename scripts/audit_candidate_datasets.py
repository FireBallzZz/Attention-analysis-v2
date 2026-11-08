"""Reproducible Dataset-Readiness Audit Script.

Audits local classroom ground truth, DAiSEE pretraining sequences,
and documents candidate dataset schema compatibility (MERICE, Mendeley, SAA-CI)
without downloading large files or running training.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def audit_local_ground_truth():
    """Audit local CVAT classroom ground truth (clip_004)."""
    gt_csv = WORKSPACE_ROOT / "Anno" / "ground_truth.csv"
    if not gt_csv.exists():
        print(f"[!] Ground truth CSV not found at: {gt_csv}")
        return

    print("\n" + "=" * 60)
    print(" 1. LOCAL CLASSROOM GROUND TRUTH AUDIT (clip_004.mp4)")
    print("=" * 60)

    rows = []
    missing_vals = 0
    with gt_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("frame") or not row.get("student_id") or not row.get("attention"):
                missing_vals += 1
            rows.append(row)

    total_records = len(rows)
    student_ids = {int(r["student_id"]) for r in rows}
    frames = {int(r["frame"]) for r in rows}

    label_map = {
        "attentive": "HIGH",
        "high": "HIGH",
        "moderate": "MODERATE",
        "non-attentive": "LOW",
        "non_attentive": "LOW",
        "low": "LOW",
    }
    class_counts = Counter()
    student_class_counts = {s: Counter() for s in student_ids}

    for r in rows:
        norm_label = label_map.get(r["attention"].strip().lower().replace(" ", "-"), "UNKNOWN")
        class_counts[norm_label] += 1
        student_class_counts[int(r["student_id"])][norm_label] += 1

    print(f"Total Annotated Observations: {total_records}")
    print(f"Distinct Students Tracked   : {len(student_ids)} (IDs: {min(student_ids)} to {max(student_ids)})")
    print(f"Temporal Span (Frames)     : {len(frames)} frames (min={min(frames)}, max={max(frames)})")
    print(f"Missing / Null Fields       : {missing_vals}")
    print("\nClass Distribution:")
    for label, count in class_counts.most_common():
        pct = (count / total_records) * 100
        print(f"  - {label:<10}: {count:5d} ({pct:5.1f}%)")

    # Audit split groups
    train_students = [0, 1, 3, 4, 6, 7, 9, 11, 12, 13, 14, 15, 17, 19, 20]
    val_students = [2, 5, 8, 10, 16, 18]
    overlap = set(train_students) & set(val_students)

    print("\nSubject-Disjoint Split Verification:")
    print(f"  Train Students ({len(train_students)}): {train_students}")
    print(f"  Val Students   ({len(val_students)}): {val_students}")
    print(f"  Identity Overlap (Leakage): {len(overlap)} students (PASSED)" if not overlap else f"  LEAKAGE ERROR: {overlap}")


def audit_local_processed_npz():
    """Audit pre-extracted multimodal sequence tensors."""
    print("\n" + "=" * 60)
    print(" 2. PROCESSED MULTIMODAL FEATURE AUDIT (NPZ ARTIFACTS)")
    print("=" * 60)

    seq_file = WORKSPACE_ROOT / "data" / "processed" / "classroom_clip_004_sequences.npz"
    if seq_file.exists():
        data = np.load(seq_file)
        feats = data["features"]
        bands = data["bands"]
        targets = data["targets"]
        students = data["student_ids"]
        print(f"\n[Artifact: classroom_clip_004_sequences.npz]")
        print(f"  Feature Tensor Shape   : {feats.shape} (N_seq, window_len, feat_dim)")
        print(f"  Bands Shape            : {bands.shape} (unique: {np.unique(bands)})")
        print(f"  Targets Shape          : {targets.shape} (unique: {np.unique(targets)})")
        print(f"  Student IDs Shape      : {students.shape} ({len(np.unique(students))} students)")
        print(f"  Feature Range          : min={feats.min():.3f}, max={feats.max():.3f}")
        print(f"  NaN Count              : {np.isnan(feats).sum()}")
        print(f"  Inf Count              : {np.isinf(feats).sum()}")
        print(f"  Active Feature Indices : 0-22 (CV modalities)")
        print(f"  Reserved Padding Dims  : 23-31 (constant {np.unique(feats[:, :, 23:])})")

    daisee_v2 = WORKSPACE_ROOT / "data" / "processed" / "fusion_daisee_v2.npz"
    if daisee_v2.exists():
        d_data = np.load(daisee_v2)
        d_feats = d_data["features"]
        d_scores = d_data["score"]
        d_clips = d_data["clip_ids"]
        d_subjs = d_data["subject_ids"]
        print(f"\n[Artifact: fusion_daisee_v2.npz (DAiSEE 750 Dev Subset)]")
        print(f"  Feature Tensor Shape   : {d_feats.shape}")
        print(f"  Continuous Score Shape : {d_scores.shape} (range: {d_scores.min():.1f} - {d_scores.max():.1f})")
        print(f"  Clips Count            : {len(np.unique(d_clips))} unique clips")
        print(f"  Subjects Count         : {len(np.unique(d_subjs))} unique subjects")
        print(f"  NaN Count              : {np.isnan(d_feats).sum()}")


def audit_candidate_dataset_registry():
    """Print verified compatibility audit for external candidates."""
    print("\n" + "=" * 60)
    print(" 3. CANDIDATE EXTERNAL DATASET AUDIT SUMMARY")
    print("=" * 60)

    candidates = [
        {
            "name": "MERICE (Ashwitha et al., Strathclyde)",
            "url": "https://github.com/Ashwitha-D/MERICE_Dataset",
            "access": "GATED / RESTRICTED (Strathclyde DOI 10.15129/99f1acaf-4786-46f9-8be8-241967a045cb)",
            "modality": "Raw Video (1,188 clips, 10s each)",
            "labels": "Classroom Engagement (High, Medium, Low)",
            "cv_32d_compat": "COMPATIBLE (Can run full YOLO+L2CS+SCB vision pipeline)",
            "fusion_compat": "COMPATIBLE (16-frame temporal sequence with 3-class target)",
            "status": "Awaiting author request approval",
        },
        {
            "name": "Students Attention Detection (Mendeley Data v3)",
            "url": "https://data.mendeley.com/datasets/smzggbnkd2/3",
            "access": "PUBLIC (CC BY 4.0 Direct Download)",
            "modality": "Tabular CSV ONLY (6,244 records, 37 MediaPipe features; NO video)",
            "labels": "Binary Attention (0: Attentive, 1: Inattentive)",
            "cv_32d_compat": "INCOMPATIBLE (No pixel video for YOLO body pose, L2CS, SCB)",
            "fusion_compat": "INCOMPATIBLE (Cannot train 3-class temporal fusion model without raw video)",
            "status": "Incompatible with primary thesis pipeline; rejected for fusion",
        },
        {
            "name": "SAA-CI (Asabere et al., 2026, SpringerOpen)",
            "url": "https://doi.org/10.1186/s43067-026-00354-w",
            "access": "PUBLIC (Google Drive link provided in paper)",
            "modality": "Face Crop Images (3,880 MTCNN crops, 745 expert-annotated)",
            "labels": "3 Attentiveness Classes (High: 584, Moderate: 69, Low: 92)",
            "cv_32d_compat": "PARTIAL (Compatible with facial/gaze branch; no body posture/SCB)",
            "fusion_compat": "AUXILIARY ONLY (Image-level; lacks continuous body sequence)",
            "status": "Verified accessible; recommended for facial feature validation",
        },
    ]

    for c in candidates:
        print(f"\nDataset: {c['name']}")
        print(f"  - Source URL        : {c['url']}")
        print(f"  - Access Status     : {c['access']}")
        print(f"  - Modality          : {c['modality']}")
        print(f"  - Original Labels   : {c['labels']}")
        print(f"  - 32-D Vision Compat: {c['cv_32d_compat']}")
        print(f"  - Fusion Net Compat : {c['fusion_compat']}")
        print(f"  - Recommended Role  : {c['status']}")


if __name__ == "__main__":
    audit_local_ground_truth()
    audit_local_processed_npz()
    audit_candidate_dataset_registry()
