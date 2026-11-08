# Phase 3: Safe DAiSEE Dataset Reconciliation Report

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis, Pedagogical Recommendation, and Learning Improvement Assessment in Smart Classrooms  
**Audit Timestamp:** 2026-10-23T15:21:00+06:00  
**Audit Phase:** Phase 3 — Dataset Reconciliation & Master Manifest Generation  

---

## 1. Executive Summary

This document reconciles all 9,068 physical video files on disk with the four official DAiSEE label CSVs and the three official text manifests.

### Core Deliverables Produced:
1. `data/manifests/daisee_master_manifest_v1.csv`: Canonical manifest of **8,986 eligible labeled video clips** with verified subject IDs, physical split partitions, affective ratings, and canonical binary attention labels.
2. `data/manifests/daisee_quarantine_manifest_v1.csv`: Quarantine manifest of **82 unannotated physical video clips** belonging to Subject `998826` in `DataSet/Test`, isolating them from supervised extraction.
3. `reports/rebuild/phase3_dataset_reconciliation.json`: Machine-readable audit ledger.

---

## 2. Master Census and Partition Reconciliation

| Partition / Split | Physical Videos on Disk | Master Labeled Clips | Quarantined Clips | Subjects | NON-ATTENTIVE (0–1) | ATTENTIVE (2–3) | Minority % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 5,482 | **5,482** | 0 | 70 | 248 | 5,234 | 4.52% |
| **Validation** | 1,720 | **1,720** | 0 | 22 | 183 | 1,537 | 10.64% |
| **Test** | 1,866 | **1,784** | 82 | 21 | 88 | 1,696 | 4.93% |
| **Total Corpus** | **9,068** | **8,986** | **82** | **113** | **519** | **8,467** | **5.78%** |

---

## 3. Discrepancy Resolution & Audit Findings

1. **Zero Conflict Union:** Comparing `AllLabels.csv` against `TrainLabels.csv`, `ValidationLabels.csv`, and `TestLabels.csv` revealed **zero** conflicting label values for any shared clip.
2. **Recovered Test Clips ($N = 61$):** `TestLabels.csv` contained 61 valid annotated clips (57 from Subject `987736` and 4 from Subject `998826`) that were omitted from `AllLabels.csv`. Incorporating these brings the verified test set to 1,784 clips.
3. **Quarantined Unlabeled Clips ($N = 82$):** Subject `998826` has 86 clips on disk in `DataSet/Test`, but only 4 are annotated in `TestLabels.csv`. The remaining 82 clips have zero official labels. Under strict dataset governance rules, no labels were imputed; all 82 clips are placed in the quarantine manifest.
4. **Subject Disjointness:** The 70 Train subjects, 22 Validation subjects, and 21 Test subjects share zero overlap ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
