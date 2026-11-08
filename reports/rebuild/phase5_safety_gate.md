# Phase 5 — Non-Destructive Extraction Safety Gate Report

**Date:** 2026-10-23  
**Auditor:** Principal Machine Learning Engineer & Reproducibility Auditor  
**Gate Decision:** **PASS (Extraction Pipeline Preflight Approved)**  

---

## 1. Executive Summary & Verification Matrix

In accordance with Section 8 of the Master Directive, a non-destructive extraction smoke test was executed using the versioned extractor [`scripts/build_fusion_v4.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_fusion_v4.py) targeting public DAiSEE samples across all three official partitions (Train, Validation, Test).

| Verification Check | Target / Invariant | Observed Result | Status |
| :--- | :--- | :--- | :--- |
| **Manifest Parsing** | Master: 8,986 clips; Quarantine: 82 clips | Exact row counts, valid columns, 0 missing video paths | **PASS** |
| **Label Contract** | Eng 0–1 $\to$ 0 (NON-ATTENTIVE); 2–3 $\to$ 1 (ATTENTIVE) | 100% compliant across all 8,986 rows | **PASS** |
| **Multi-Split Representation** | 1 clip from Train, Val, Test | Clips `1100011002` (Train), `4000221001` (Val), `5000441001` (Test) | **PASS** |
| **Feature Tensor Geometry** | Shape: $(N, 16, 32)$, dtype: `float32` | Shape: `(71, 16, 32)`, dtype: `float32` | **PASS** |
| **Metadata Array Alignment** | Identical length $N=71$ across all fields | `features`, `score`, `aux`, `clip_ids`, `subject_ids`, `splits`, `binary_targets`, `frame_indices` all $N=71$ | **PASS** |
| **Numerical Integrity** | No NaN, no Inf, finite float values | Zero NaNs, zero Infs across all 71 windows | **PASS** |
| **Atomic Save Mechanics** | Temporary file `.tmp.npz` $\to$ atomic replace | Verified in isolation and during extraction | **PASS** |
| **Interruption & Resume** | Process 2 clips $\to$ exit $\to$ resume 3rd clip | Loaded 2 clips, resumed seamlessly, processed 3rd clip; total 71 seqs | **PASS** |
| **Idempotence** | Re-running on completed archive | Detected 3/3 done, processed 0, zero duplicate windows | **PASS** |
| **Downstream Compatibility** | `ManualBinaryAttentionClassifier` forward pass | Forward pass succeeded; outputs valid probability distributions | **PASS** |
| **Protected Assets Integrity** | Bit-for-bit SHA-256 match on 6 protected files | All 6 files match baseline hashes exactly | **PASS** |
| **Forbidden Paths Exclusion** | Zero access to `YOLO Test.mp4` and UAP videos | Verified excluded from search paths and manifests | **PASS** |

---

## 2. Test Execution Details

### 2.1 Smoke Test Multi-Split Extraction & Resume Test
1. **Initial Run:** Extracted clips `1100011002` (Train) and `4000221001` (Validation) into [`data/processed/smoke_test_fusion.npz`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/processed/smoke_test_fusion.npz).
   - Produced 36 sequences (18 per clip, 16-frame sliding window with 50% overlap).
2. **Simulated Interruption & Resume:** Invoked [`scripts/build_fusion_v4.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_fusion_v4.py) targeting all 3 clips.
   - Identified 2 completed clips from the existing archive.
   - Closed file handles to prevent Windows filesystem lockups.
   - Extracted remaining clip `5000441001` (Test).
   - Atomically replaced archive via `.tmp.npz` $\to$ `.npz`.
   - Result: 71 sequences across 3 clips (35 sequences for clip 5000441001 due to duration).
3. **Idempotence Verification:** Re-ran extractor targeting all 3 clips.
   - Output: `Targeting 3 clips ... Resuming: smoke_test_fusion.npz already has 3 clips completed ... Clips remaining to process: 0 / 3`.
   - Exited immediately with zero changes and zero duplicated sequences.

### 2.2 Downstream Model Forward Pass
The extracted tensor `(71, 16, 32)` was passed through the active production model architecture ([`ManualBinaryAttentionClassifier`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/src/attention_pipeline/models/manual_binary.py)):
- Input: `torch.Size([71, 16, 32])`
- Logits Output: `torch.Size([71, 2])`
- Output Probabilities: `probs[:, 1]` in range $[0.99956, 0.99999]$ (attentive predictions corresponding to engagement score 2)
- Scores: Range $[99.96, 100.00]$

### 2.3 Unit Test Suite
The automated test suite in [`tests/test_phase8a_rebuild.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/tests/test_phase8a_rebuild.py) was expanded to 6 comprehensive tests:
- `test_master_manifest_integrity`: PASSED
- `test_quarantine_manifest_integrity`: PASSED
- `test_split_subject_disjointness`: PASSED
- `test_atomic_save_mechanism`: PASSED
- `test_protected_checkpoints_invariant`: PASSED
- `test_smoke_archive_schema_and_downstream_model`: PASSED
- Overall test suite: **72 passed, 1 skipped in 4.88s**.

---

## 3. Full Public-Data Extraction Specifications (Phase 6 / Gate D)

### 3.1 Resource & Runtime Analysis
- **Target Dataset:** Public DAiSEE Corpus via [`data/manifests/daisee_master_manifest_v1.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/manifests/daisee_master_manifest_v1.csv).
- **Target Count:** 8,986 clips (5,482 Train, 1,720 Validation, 1,784 Test).
- **Hardware:** NVIDIA GeForce RTX 3060 (12GB VRAM), Intel Core i7-11700F, 32GB RAM.
- **Measured Throughput:** ~4.30 seconds / clip (~0.23 clips / second).
- **Total Estimated Wall-Clock Runtime:** **10.74 hours** (approx. 38,640 seconds).
  - Train split (5,482 clips): ~6.55 hours
  - Validation split (1,720 clips): ~2.06 hours
  - Test split (1,784 clips): ~2.13 hours
- **Disk Storage Requirement:**
  - 8,986 clips $\approx$ 160,000–210,000 sequences $\approx$ 450–600 MB compressed NPZ.
  - Available disk space: **400.3 GB free** (ample capacity).

### 3.2 Canonical Launch Command
```bash
python scripts/build_fusion_v4.py \
    --manifest data/manifests/daisee_master_manifest_v1.csv \
    --out-npz data/processed/fusion_daisee_full_v4.npz \
    --fps 5.0 \
    --save-interval 100
```

### 3.3 Protection & Recovery Features
1. **Periodic Resumable Checkpoints:** Every 100 clips, the archive is updated atomically. If interrupted (power loss, system sleep, or user break), rerunning the exact command will resume from clip $N+1$ without loss or duplication.
2. **Protected Checkpoint Isolation:** No protected model or pre-existing NPZ (`fusion_daisee.npz`, `fusion_daisee_v2.npz`, `fusion_daisee_v3_hardened.npz`) will be modified or overwritten.
3. **Partitioned Option:** If preferred, extraction can be run split-by-split using the `--split` flag:
   - `python scripts/build_fusion_v4.py --split Validation ...` (2.06 hours)
   - `python scripts/build_fusion_v4.py --split Test ...` (2.13 hours)
   - `python scripts/build_fusion_v4.py --split Train ...` (6.55 hours)

---

## 4. Protected Asset Hash Verification (Gate C Post-Check)

| Asset Path | Baseline SHA-256 | Gate C Current SHA-256 | Verification |
| :--- | :--- | :--- | :--- |
| `checkpoints/classroom_manual_binary_model_v2_clean.pt` | `2BFA5AF8...42124E` | `2BFA5AF8...42124E` | **MATCH (VERIFIED)** |
| `checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt` | `E01F33BF...3F26F8E` | `E01F33BF...3F26F8E` | **MATCH (VERIFIED)** |
| `checkpoints/candidates/phase7_public_trained_binary_candidate.pt` | `1847B2B9...0EDAE5` | `1847B2B9...0EDAE5` | **MATCH (VERIFIED)** |
| `data/processed/fusion_daisee.npz` | `15B3FB28...B5196B` | `15B3FB28...B5196B` | **MATCH (VERIFIED)** |
| `data/processed/fusion_daisee_v2.npz` | `69E5C998...EE76733` | `69E5C998...EE76733` | **MATCH (VERIFIED)** |
| `data/processed/fusion_daisee_v3_hardened.npz` | `4505FB44...FD2DA6` | `4505FB44...FD2DA6` | **MATCH (VERIFIED)** |

**Conclusion:** Safety Gate C is **PASSED**. Full public-data feature extraction (Phase 6 / Gate D) is authorized.
