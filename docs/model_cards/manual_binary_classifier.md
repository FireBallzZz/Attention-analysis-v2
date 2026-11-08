# Model Card: ManualBinaryAttentionClassifier

**Model Name:** `ManualBinaryAttentionClassifier`  
**Checkpoint:** `checkpoints/classroom_manual_binary_model_v2_clean.pt`  
**SHA-256 Hash:** `2BFA5AF8E190B5D4C16D463510543DBDCA2D8FCEF43CFDC1AE10E8C46C42124E`  
**Architecture:** AttentionFusionNet (Frozen Linear Projection + Bidirectional GRU) + Multi-Pooling Head (Last + Mean + Max = 384-D) + 2-layer MLP  

---

## 1. Intended Use

- **Task:** Temporal binary student attention classification (Class 0: `NON-ATTENTIVE`, Class 1: `ATTENTIVE`) and continuous attention scoring ($0.0 - 100.0$).
- **Input:** 16-frame sequence of 32-D multimodal feature vectors extracted at 5.0 FPS (3.2 seconds sliding window).
- **Output:** Logits $\in \mathbb{R}^2$, Softmax probabilities $[P(\text{LOW}), P(\text{HIGH})]$, Continuous attention score $P(\text{HIGH}) \times 100.0$.
- **Temporal Warmup:** Requires 16 consecutive frames before emitting predictions. Returns `band="WARMUP"` with score `None` during frames 1..15.

---

## 2. Training & Tuning Data

- **Pretraining:** Backbone initialized from `checkpoints/fusion_daisee.pt` trained on public DAiSEE multimodal feature sequences.
- **Head Tuning:** Manually calibrated and hardened against classroom visual behavioral rubrics to preserve sensitivity to genuine non-attentive postures (bowing head, sleeping, looking away) despite severe public dataset class imbalance.

---

## 3. Evaluation on Held-Out Public Test Set

Evaluated on the frozen, subject-disjoint DAiSEE v3 hardened test split (subjects `110001` and `110004`, 148 clips, 3,563 sequences):

| Metric | Value @ Default $\tau = 0.50$ | Value @ Calibrated $\tau^* = 0.10$ |
| :--- | :--- | :--- |
| **Accuracy** | 48.19% | 71.71% |
| **Balanced Accuracy (95% CI)** | 49.08% [21.26%, 76.99%] | 59.67% [34.07%, 85.62%] |
| **Macro-F1 (95% CI)** | 34.11% [29.17%, 39.67%] | 44.74% [40.08%, 51.65%] |
| **Non-Attentive (0) Recall** | **50.00%** (35/70 identified) | **47.14%** (33/70 identified) |
| **Non-Attentive (0) Precision** | 1.90% | 3.29% |
| **Attentive (1) Recall** | 48.15% (1,682/3,493) | 72.20% (2,522/3,493) |
| **Attentive (1) Precision** | 97.96% | 98.55% |
| **Confusion Matrix $[[\text{TN}, \text{FP}], [\text{FN}, \text{TP}]]$** | $[[35, 35], [1811, 1682]]$ | $[[33, 37], [971, 2522]]$ |
| **Brier Score** | 0.2858 | 0.2858 |
| **ROC-AUC** | 0.6279 | 0.6279 |
