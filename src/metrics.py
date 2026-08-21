"""Metrics for event-group probes, codebook health, and implementation cost.

The benchmark reports classification metrics, majority-class baseline accuracy,
relative probe accuracy, codebook utilization/perplexity, nominal discrete
index rates, latency, and parameter counts where applicable.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n": 0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "n": int(len(y_true)),
    }


def confusion(y_true, y_pred, n_labels):
    labels = list(range(n_labels))
    return confusion_matrix(np.asarray(y_true), np.asarray(y_pred), labels=labels)


def majority_class_baseline_accuracy(y) -> float:
    """Return the accuracy of always predicting the most frequent class."""
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def relative_probe_accuracy(acc_representation: float, acc_reference: float) -> float:
    """Return representation probe accuracy divided by reference probe accuracy.

    The reference is typically pooled log-mel. This ratio can exceed one and is
    not a bounded fraction of mutual information.
    """
    if acc_reference <= 0:
        return 0.0
    return float(acc_representation / acc_reference)


def codebook_utilization(indices, codebook_size: int):
    """indices: any array of integer code ids. Returns usage + perplexity."""
    idx = np.asarray(indices).reshape(-1)
    counts = np.bincount(idx, minlength=codebook_size).astype(np.float64)
    total = max(counts.sum(), 1.0)
    p = counts / total
    nz = p[p > 0]
    entropy = float(-(nz * np.log2(nz)).sum()) if len(nz) else 0.0
    return {
        "codebook_size": int(codebook_size),
        "codes_used": int((counts > 0).sum()),
        "utilization": float((counts > 0).sum() / codebook_size),
        "entropy_bits": entropy,
        "perplexity": float(2.0 ** entropy),
    }
