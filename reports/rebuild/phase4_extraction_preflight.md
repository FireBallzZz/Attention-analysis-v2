# Phase 4: Feature-Extraction Pipeline Rebuild & Preflight Report

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Audit Timestamp:** 2026-10-23T15:23:00+06:00  
**Audit Phase:** Phase 4 — Extraction Pipeline Rebuild (v4)  

---

## 1. Executive Summary

To resolve the limitation in legacy extraction scripts (which hardcoded a 750-clip source archive), a new versioned extractor was developed:
[`scripts/build_fusion_v4.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_fusion_v4.py).

The legacy script [`scripts/build_fusion_v3.py`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/scripts/build_fusion_v3.py) and historical datasets (`fusion_daisee.npz`, `fusion_daisee_v2.npz`, `fusion_daisee_v3_hardened.npz`) remain **100% untouched and preserved**.

---

## 2. Key Technical Improvements in `build_fusion_v4.py`

1. **Manifest-Driven Targeting:**
   Directly parses [`data/manifests/daisee_master_manifest_v1.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/manifests/daisee_master_manifest_v1.csv), targeting all **8,986 eligible labeled clips** and strictly excluding the 82 unannotated clips of Subject `998826`.
2. **Atomic Replacement Architecture:**
   Writes compressed NPZ archives to `.tmp.npz` before replacing destination via `tmp_path.replace(out_path)`. This guarantees zero archive corruption if process is killed or interrupted mid-compression.
3. **Resumability Verification:**
   Checks completed clips via `prev["clip"]` on startup; resumes seamlessly from the last saved boundary without duplicating sequences or omitting clips.
4. **Enriched Output Schema:**
   In addition to standard arrays (`features`, `score`, `aux`, `clip_ids`, `subject_ids`, `frame_indices`, `clip`), `v4` stores:
   - `splits`: String array indicating partition (`Train`, `Validation`, `Test`).
   - `binary_targets`: Int64 array adhering strictly to canonical binary attention contract ($0 = \\text{NON-ATTENTIVE}$, $1 = \\text{ATTENTIVE}$).
5. **Phase 6B/6C Hardened Feature Parity:**
   Maintains torso lean inter-shoulder tilt fallback (Channel 22) and upper-body keypoint confidence (Channel 23).
