"""Frozen-feature linear probe (logistic regression).

A linear probe on frozen features asks:
"how much task-relevant information is linearly decodable from this
representation?" We use it identically on raw log-mel features and on
codec-token features, while reporting that their dimensions and pretraining
conditions differ.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def train_probe(X_train, y_train, C=1.0, max_iter=2000, seed=0):
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    clf.fit(scaler.transform(X_train), y_train)
    return clf, scaler


def predict(clf, scaler, X):
    return clf.predict(scaler.transform(X))


def crossval(X, y, folds, C=1.0, max_iter=2000, seed=0):
    """ESC-50-style cross-validation over integer fold ids.

    Returns dict with per-fold accuracy, mean/std, and pooled (y_true, y_pred)
    aggregated across held-out folds for a single confusion matrix.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    folds = np.asarray(folds)
    accs = []
    all_true, all_pred = [], []
    for f in sorted(np.unique(folds)):
        te = folds == f
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        clf, scaler = train_probe(X[tr], y[tr], C=C, max_iter=max_iter, seed=seed)
        yp = predict(clf, scaler, X[te])
        accs.append(float((yp == y[te]).mean()))
        all_true.append(y[te])
        all_pred.append(yp)
    accs = np.array(accs, dtype=np.float64)
    return {
        "fold_accuracy": accs.tolist(),
        "mean_accuracy": float(accs.mean()) if len(accs) else 0.0,
        "std_accuracy": float(accs.std()) if len(accs) else 0.0,
        "y_true": np.concatenate(all_true) if all_true else np.array([]),
        "y_pred": np.concatenate(all_pred) if all_pred else np.array([]),
    }
