# Phase 6 Extraction Pre-Launch Readiness Report

**Date:** 2026-10-23  
**Auditor:** Reproducibility Engineer & ML Systems Architect  
**Gate Decision:** **PASS (Extraction Launch Authorized)**  

---

## 1. Pre-Launch Verification Matrix

| Verification Item | Target Standard | Observed Status | Gate Result |
| :--- | :--- | :--- | :---: |
| **Extractor Source Code** | Versioned [`scripts/build_fusion_v4.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_fusion_v4.py) | Atomic `.tmp.npz` write, resume safe, handle closed | **PASS** |
| **Master Manifest Integrity** | 8,986 eligible labeled clips | 8,986 rows; exact matching physical video paths | **PASS** |
| **Quarantine Manifest Integrity** | 82 unannotated clips of subject `998826` | 82 rows; completely isolated; 0 inferred labels | **PASS** |
| **Split Disjointness** | 0% subject overlap across Train/Val/Test | 70 Train, 22 Val, 21 Test subjects; 0 overlap | **PASS** |
| **Disk Space Availability** | Free space $> 10$ GB | **400.67 GB free** (ample capacity) | **PASS** |
| **Protected Checkpoints** | Bit-for-bit SHA-256 match (3 checkpoints) | All 3 match baseline hashes exactly | **PASS** |
| **Protected Datasets** | Bit-for-bit SHA-256 match (3 NPZ archives) | All 3 match baseline hashes exactly | **PASS** |
| **Forbidden Assets Exclusion** | `YOLO Test.mp4` & UAP classroom videos | Excluded from manifest and extractor paths | **PASS** |
| **Destination Output Path** | Unused path `data/processed/fusion_daisee_full_v4.npz` | Path verified non-existent; ready for clean creation | **PASS** |
| **Interruption & Resume** | Idempotent resume without duplicate windows | Smoke test confirmed 100% idempotent resume | **PASS** |

---

## 2. Launch Execution Configuration

- **Command Line:**
  ```bash
  python scripts/build_fusion_v4.py \
      --manifest data/manifests/daisee_master_manifest_v1.csv \
      --out-npz data/processed/fusion_daisee_full_v4.npz \
      --fps 5.0 \
      --save-interval 50
  ```
- **Logging Strategy:** Persistent log stream to `runs/logs/extraction_v4.log` with real-time progress, ETA, and clip rate reporting.
- **Estimated Wall-Clock Runtime:** 10.74 hours @ 4.30s/clip across 8,986 clips.
- **Checkpointing Interval:** Every 50 clips (~3.5 minutes), atomic replacement of destination `.npz`.
