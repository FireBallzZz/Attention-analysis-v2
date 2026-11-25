"""Correct binary metrics for attention evaluation (class 1 = HIGH, class 0 = LOW).

Earlier scripts computed F1 from the HIGH class only and labelled it
"macro F1". With ~85% HIGH windows that hides a model that never finds LOW.
Every report should use `binary_report` and always show the always-HIGH
baseline next to accuracy.
"""

from __future__ import annotations


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def binary_report(tn: int, fp: int, fn: int, tp: int) -> dict:
    """All values in percent. Rows: true LOW = (tn, fp), true HIGH = (fn, tp)."""
    n = tn + fp + fn + tp
    prec_hi = tp / (tp + fp) if tp + fp else 0.0
    rec_hi = tp / (tp + fn) if tp + fn else 0.0
    prec_lo = tn / (tn + fn) if tn + fn else 0.0
    rec_lo = tn / (tn + fp) if tn + fp else 0.0
    f1_hi, f1_lo = _f1(prec_hi, rec_hi), _f1(prec_lo, rec_lo)
    pct = lambda v: round(100.0 * v, 2)  # noqa: E731
    return {
        "n": n,
        "accuracy": pct((tp + tn) / n) if n else 0.0,
        "balanced_accuracy": pct((rec_hi + rec_lo) / 2),
        "macro_f1": pct((f1_hi + f1_lo) / 2),
        "high_precision": pct(prec_hi),
        "high_recall": pct(rec_hi),
        "high_f1": pct(f1_hi),
        "low_precision": pct(prec_lo),
        "low_recall": pct(rec_lo),
        "low_f1": pct(f1_lo),
        "always_high_accuracy": pct((tp + fn) / n) if n else 0.0,
        "confusion": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
    }
