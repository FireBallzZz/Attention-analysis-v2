"""Hash Generator and Inventory Collector for Audit Artifacts.
Computes SHA256 hashes, file sizes, and existence checks.
"""

import hashlib
import json
from pathlib import Path

WORKSPACE_ROOT = Path("C:/Users/User/Downloads/Thesis/Model/Attention-analysis-model")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_inventory():
    key_files = [
        # Checkpoints
        ("checkpoints/classroom_manual_binary_unquarantined.pt", "Champion production checkpoint (BiGRU + MultiPool)", True),
        ("checkpoints/scb_yolo.pt", "SCB-6 classroom behavior detection weights", True),
        ("checkpoints/l2cs_finetuned.pt", "L2CS gaze estimation weights", True),
        ("checkpoints/baseline/fusion_daisee.pt", "DAiSEE pre-trained temporal GRU backbone", True),
        ("yolo11n-pose.pt", "YOLO11-Pose detector (input: 1920px)", True),
        # Source code
        ("run_production_pipeline.py", "Production video inference entrypoint", True),
        ("src/attention_pipeline/pipeline.py", "Core multi-stage attention pipeline", True),
        ("src/attention_pipeline/stages/detect_track.py", "YOLO11 pose & ByteTrack multi-object tracking", True),
        ("src/attention_pipeline/features.py", "32-D multimodal feature packer and geometry extractors", True),
        ("src/attention_pipeline/rubric.py", "Classroom behavioral heuristic rubric & action priors", True),
        ("src/attention_pipeline/models/attention_fusion.py", "Base AttentionFusionNet (frame encoder + BiGRU)", True),
        ("src/attention_pipeline/models/manual_binary.py", "ManualBinaryAttentionClassifier (Multi-Pool 384-D)", True),
        ("src/attention_pipeline/video_accuracy.py", "Video accuracy reporting & heuristic verification", True),
        # Configs
        ("configs/pipeline.yaml", "Primary pipeline hyperparameter configuration", True),
        # Training & Tuning
        ("scripts/tune_classroom_model_scientific.py", "Grid search and model tuning script", True),
        ("scripts/train_classroom_binary_unquarantined.py", "Training script for unquarantined binary classifier", True),
        ("scripts/extract_classroom_sequences.py", "Feature sequence extractor from classroom clips", True),
        # Annotations & Data
        ("runs/Annoted file for clip 1/annotations.xml", "User CVAT human annotation for Clip 001", True),
        ("Anno/annotations.xml", "CVAT ground truth bounding boxes for Clip 004/007/008", True),
        ("Anno/ground_truth.csv", "CVAT manual attention ground-truth labels", True),
        ("data/processed/classroom_clip_001_v3_hardened.npz", "Clip 001 extracted feature sequences", True),
        ("data/processed/classroom_clip_004_v3_hardened.npz", "Clip 004 extracted feature sequences", True),
        ("data/processed/classroom_clip_008_v3_hardened.npz", "Clip 008 extracted feature sequences", True),
        # Prior reports & logs
        ("runs/reports/Student_Attention_Model_Supervisor_Report.pdf", "Prior supervisor report (audited artifact)", False),
        ("runs/reports/classroom_unquarantined_binary_metrics.json", "Evaluation log for unquarantined training run", False),
        ("runs/reports/tuning_scientific_audit.json", "Hyperparameter grid search tuning log", False),
        ("runs/reports/clip_001_metrics.json", "Clip 001 evaluation log", False),
        ("runs/reports/clip_007_metrics.json", "Clip 007 evaluation log", False),
        ("runs/reports/video_project_3_metrics.json", "Video Project 3 evaluation log", False),
        ("runs/reports/test_4_metrics.json", "Test 4 evaluation log", False),
        # Videos
        ("runs/clip_001.mp4", "Classroom clip 001 (10s benchmark)", True),
        ("runs/clip_007.mp4", "Classroom clip 007 (10s benchmark)", True),
        ("runs/Video Project 3.mp4", "Large lecture hall test video (23s)", True),
        ("runs/test 4.mp4", "Multi-row classroom test video (60s)", True),
    ]

    inventory = []
    for rel_path, purpose, required in key_files:
        p = WORKSPACE_ROOT / rel_path
        exists = p.exists()
        size_bytes = p.stat().st_size if exists else 0
        sha = sha256_file(p) if exists and size_bytes < 100_000_000 else ("<computed on demand / large file>" if exists else "N/A")
        inventory.append({
            "path": rel_path,
            "purpose": purpose,
            "exists": exists,
            "required_for_reproduction": required,
            "size_bytes": size_bytes,
            "sha256": sha,
        })

    out_file = WORKSPACE_ROOT / "audit" / "FILE_INVENTORY.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"Generated inventory for {len(inventory)} artifacts.")

if __name__ == "__main__":
    get_inventory()

