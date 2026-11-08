# Dataset Governance, Manifest Integrity, and Binary Label Policy

**Project:** Hybrid Machine Learning and Large Language Model Framework for Student Attention Analysis in Smart Classrooms.  
**Standard:** Phase 8 Scientific Rigor & Dataset Governance Policy  

---

## 1. Public DAiSEE Corpus Reconciliation

### 1.1 Dataset Provenance & Master Union
- **Source:** Official DAiSEE (Dataset for Affective States in E-Environments) release by IIT Mandi.
- **Root Directory:** Located and validated at `C:\Users\User\Downloads\Thesis\PRE DEFENCE\DAiSEE`.
- **Master Manifest:** [`data/manifests/daisee_master_manifest_v1.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/manifests/daisee_master_manifest_v1.csv) contains **8,986 eligible labeled clips** constructed by reconciling official CSVs (`AllLabels.csv`, `TrainLabels.csv`, `ValidationLabels.csv`, `TestLabels.csv`).
  - Train Partition: 5,482 clips (70 subjects).
  - Validation Partition: 1,720 clips (22 subjects).
  - Test Partition: 1,784 labeled clips (21 subjects).

### 1.2 Unlabeled Quarantine Isolation
- **Quarantined Corpus:** 82 physical video files located in `DataSet/Test/998826` are completely unannotated in all four official label CSVs.
- **Policy:** In strict adherence to scientific integrity, labels were **never inferred** from filenames, visual inspection, or synthetic heuristics.
- **Quarantine Manifest:** All 82 clips are cataloged and quarantined in [`data/manifests/daisee_quarantine_manifest_v1.csv`](file:///c:/Users/User/Downloads/Thesis/Model/Attention-analysis-model/data/manifests/daisee_quarantine_manifest_v1.csv) and excluded from supervised feature extraction and model training.

### 1.3 Subject-Level Split Disjointness
- All partitions are 100% subject-disjoint:
  - $\text{Train} \cap \text{Validation} = \emptyset$ (0 subjects overlap)
  - $\text{Train} \cap \text{Test} = \emptyset$ (0 subjects overlap)
  - $\text{Validation} \cap \text{Test} = \emptyset$ (0 subjects overlap)
- Windows are clustered by physical video clip. Random cross-sequence splitting is strictly prohibited to prevent leakage of identical consecutive frames across training and validation sets.

---

## 2. Canonical Binary Attention Contract

The operational classification target is defined as:

$$\text{Class } 0 = \text{NON-ATTENTIVE} \quad (\text{DAiSEE Engagement Levels } 0 \text{ or } 1)$$
$$\text{Class } 1 = \text{ATTENTIVE} \quad (\text{DAiSEE Engagement Levels } 2 \text{ or } 3)$$

### Invariants:
1. **Never Reverse Class Order:** All confusion matrices, checkpoints, metrics, and probability outputs strictly adhere to `Index 0: Non-Attentive`, `Index 1: Attentive`.
2. **Metadata Preservation:** Original continuous regression scores ($0..3$) for Engagement, Boredom, Confusion, and Frustration are preserved in sidecar metadata and aux targets.
3. **Operational Surrogate Notice:** Attention is operationalized based on observable behavioral engagement annotations. It does not measure internal cognitive brain states.

---

## 3. Forbidden Assets & Quarantined Data

1. **`PRE DEFENCE/YOLO Test.mp4`**: Strictly forbidden. Never opened, read, decoded, or processed by any pipeline script.
2. **UAP Classroom Footage (`Clip 001.mp4`, `Clip 004.mp4`, `Clip 008.mp4`)**:
   - Quarantined exclusively for final frozen external-domain evaluation.
   - Never used for feature selection, hyperparameter tuning, model training, threshold selection, or early stopping.
