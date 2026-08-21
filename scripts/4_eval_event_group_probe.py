"""Evaluate frozen-probe performance for one or more feature files.

Loads features_*.npz produced by scripts 2, 3, and 6, runs the SAME
linear-probe cross-validation on each, and reports absolute accuracy/macro-F1
plus a secondary relative-accuracy ratio versus the pooled log-mel reference.

By default only real ESC-50 feature files are selected. Synthetic self-test
artifacts are never mixed into final aggregation unless explicitly requested.

    python scripts/4_eval_event_group_probe.py
    python scripts/4_eval_event_group_probe.py --features assets/results/features_mel_probe_esc50.npz assets/results/features_encodec_esc50.npz
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src import probe as P
from src import metrics as M
from src.visualize import plot_confusion


def representation_name(path):
    name = os.path.basename(path)
    if "features_encodec_" in name:
        return "EnCodec (first-2 codebook prefix)"
    if "features_mel_probe_" in name:
        return "Log-mel (mean + std)"
    if "features_melrvq_" in name:
        return "Custom mel-RVQ (first-2 codebook prefix)"
    return name.replace("features_", "").replace(".npz", "")


def evaluate_npz(path, cfg):
    d = np.load(path, allow_pickle=False)
    X, y, folds = d["X"], d["y"], d["folds"]
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError(
            f"{path} has fewer than two folds ({unique_folds.tolist()}); "
            "this is not a valid cross-validation artifact.")
    cv = P.crossval(X, y, folds, C=cfg["probe_C"], max_iter=cfg["probe_max_iter"],
                    seed=cfg["seed"])
    if not cv["fold_accuracy"]:
        raise ValueError(f"{path} produced no held-out fold predictions.")
    cls = M.classification_metrics(cv["y_true"], cv["y_pred"])
    labels = [str(x) for x in d["group_names"].tolist()]
    cm = M.confusion(cv["y_true"], cv["y_pred"], n_labels=len(labels))
    row = {
        "file": os.path.basename(path),
        "representation": representation_name(path),
        "feature_dim": int(X.shape[1]),
        "n_clips": int(X.shape[0]),
        "folds": [int(x) for x in unique_folds.tolist()],
        "mean_accuracy": cv["mean_accuracy"],
        "std_accuracy": cv["std_accuracy"],
        "macro_f1": cls["macro_f1"],
        "majority_class_baseline_accuracy": M.majority_class_baseline_accuracy(y),
        "fold_accuracy": cv["fold_accuracy"],
    }
    return row, cm, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--source", default="esc50",
                    help="feature filename suffix to aggregate (default: esc50; use all explicitly)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    outdir = args.outdir or cfg["results_dir"]

    if args.features:
        files = args.features
    elif args.source == "all":
        files = sorted(glob.glob(os.path.join(outdir, "features_*.npz")))
    else:
        files = sorted(glob.glob(os.path.join(outdir, f"features_*_{args.source}.npz")))
    if not files:
        print("No feature files found. Run scripts 2, 3, and/or 6 first.")
        sys.exit(2)

    evaluated = [evaluate_npz(f, cfg) for f in files]
    rows = [x[0] for x in evaluated]
    # Select the pooled log-mel reference exactly, never mel-RVQ.
    ref = next((r for r in rows if r["file"].startswith("features_mel_probe_")), None)
    ref_acc = ref["mean_accuracy"] if ref else None
    for r in rows:
        r["relative_probe_accuracy_vs_logmel"] = (
            M.relative_probe_accuracy(r["mean_accuracy"], ref_acc) if ref_acc else None)

    os.makedirs(cfg["figures_dir"], exist_ok=True)
    for row, cm, labels in evaluated:
        stem = os.path.splitext(row["file"].replace("features_", ""))[0]
        figure = os.path.join(cfg["figures_dir"], f"confusion_{stem}.png")
        plot_confusion(cm, labels, figure, title=row["representation"])

    print(f"{'representation':42s} {'acc':>7s} {'F1':>7s} {'relative':>10s}")
    print("-" * 70)
    for r in rows:
        relative = r["relative_probe_accuracy_vs_logmel"]
        rel_text = "-" if relative is None else f"{relative:.2f}"
        print(f"{r['representation']:42s} {r['mean_accuracy']:7.3f} "
              f"{r['macro_f1']:7.3f} {rel_text:>10s}")

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"event_group_probe_summary_{args.source}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "source": args.source,
            "primary_metrics": ["mean_accuracy", "macro_f1"],
            "relative_metric": "probe accuracy divided by pooled log-mel accuracy; may exceed 1.0",
            "reference_accuracy": ref_acc,
            "rows": rows,
        }, f, indent=2)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
