# Reproducibility, Verification, and Environmental Specifications

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis in Smart Classrooms.  
**Auditor:** Reproducibility Engineer & Systems Architect  

---

## 1. System and Hardware Environment

- **Operating System:** Windows 11 Enterprise (Build 10.0.26100)
- **CPU:** Intel Core i7-11700F @ 2.50GHz (8 cores, 16 threads)
- **System Memory:** 32.0 GB RAM
- **GPU:** NVIDIA GeForce RTX 3060 (12,288 MB VRAM)
- **CUDA Runtime:** 12.6
- **Python Version:** 3.14.4
- **PyTorch Version:** 2.14.1+cu126
- **TorchVision Version:** 0.20.1+cu126
- **Available Disk Space:** 400.3 GB free

---

## 2. Protected Checkpoint & Dataset Verification Baselines

All 6 protected assets are verified bit-for-bit using cryptographic SHA-256 hashes before and after every pipeline stage. A hash discrepancy represents an absolute STOP condition.

| Asset Path | Exact SHA-256 Hash | Size (Bytes) | Governance Role |
| :--- | :--- | :--- | :--- |
| `checkpoints/classroom_manual_binary_model_v2_clean.pt` | `2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E` | 466,837 | Active Production Model |
| `checkpoints/candidates/phase6g4c_engaged_disengaged_candidate.pt` | `E01F33BF7859A7A06035D17DF60BA7B406DCD8307A8F5EE8F98EE86A63F26F8E` | 467,993 | Phase 6 Candidate Checkpoint |
| `checkpoints/candidates/phase7_public_trained_binary_candidate.pt` | `1847B2B9683EC40632FBAEC3B7191592305EBECEFAE855352C8E81C2230EDAE5` | 468,313 | Phase 7 Public Candidate Checkpoint |
| `data/processed/fusion_daisee.npz` | `15B3FB28EF15AAC1564423251F36825A5BC9FBB5E2F0353D27D8978AACB5196B` | 48,335,334 | Legacy DAiSEE Full Extraction |
| `data/processed/fusion_daisee_v2.npz` | `69E5C9984A27980AAC36BA14F190534D84AE5260145A114502D278627EE76733` | 1,479,987 | DAiSEE Hardened v2 Dataset |
| `data/processed/fusion_daisee_v3_hardened.npz` | `4505FB4428944366459B1E3AEE37678A8AD7381AA19384A64514C87F75FD2DA6` | 1,421,537 | Audited Subject-Disjoint Dataset |

---

## 3. Seed Control & Determinism

- **Random Seeds:** Pinned to `seed = 42` across Python `random`, NumPy `np.random.seed(42)`, PyTorch `torch.manual_seed(42)`, and CUDA `torch.cuda.manual_seed_all(42)`.
- **CuDNN Flags:** `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.

---

## 4. Execution Commands for Pipeline Reproduction

### 4.1 Verification & Unit Testing
```bash
python -m pytest tests/test_phase8a_rebuild.py -v
python -m pytest tests -q
```

### 4.2 Smoke Test Feature Extraction & Resume
```bash
python scripts/build_fusion_v4.py \
    --manifest data/manifests/smoke_test_manifest.csv \
    --out-npz data/processed/smoke_test_fusion.npz \
    --save-interval 1
```

### 4.3 Full Public DAiSEE Extraction (10.74h Wall-Clock)
```bash
python scripts/build_fusion_v4.py \
    --manifest data/manifests/daisee_master_manifest_v1.csv \
    --out-npz data/processed/fusion_daisee_full_v4.npz \
    --fps 5.0 \
    --save-interval 100
```

### 4.4 Controlled Training, Validation, and Frozen Test Evaluation
```bash
python scripts/train_phase8_experiments.py --epochs 25 --patience 6 --batch-size 128
```

### 4.5 Production Inference Smoke Test
```bash
python -m attention_pipeline "C:\Users\User\Downloads\Thesis\PRE DEFENCE\DAiSEE\DataSet\Train\110001\1100011002\1100011002.avi" \
    --seconds 5 \
    --out runs/experiments/phase8/inference_smoke.jsonl \
    --out-video runs/experiments/phase8/inference_overlay_smoke.mp4
```
