"""FIRST RUNNABLE BASELINE -- raw log-mel + linear probe (zero extra deps).

This is the pooled log-mel reference interface. Every codec/tokenizer is later
compared against its probe score via the secondary relative-accuracy ratio; it
is not an information-theoretic upper bound.

Runs with only packages already in the environment (torch, torchaudio,
numpy, scipy, scikit-learn, pandas, matplotlib).

Quick proof it works (no download, ~seconds):
    python scripts/2_mel_probe_baseline.py --self-test

Real run on ESC-50 (after scripts/1_prepare_data.py):
    python scripts/2_mel_probe_baseline.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src import event_groups as eg
from src.features import LogMel, pool_stats, resolve_feature_device
from src import probe as P
from src import metrics as M
from src.visualize import plot_confusion


def build_features(items, cfg, device, log_every=200):
    """items: list of dicts {wav, sr, group_index, fold}. Returns X, y, folds."""
    lm = LogMel(sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
                f_min=cfg["f_min"], f_max=cfg["f_max"], device=device)
    X, y, folds = [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        mel = lm.wave_to_logmel(it["wav"], it["sr"])
        X.append(pool_stats(mel))
        y.append(it["group_index"])
        folds.append(it["fold"])
        if log_every and (i + 1) % log_every == 0:
            print(f"  features {i+1}/{len(items)}  ({time.time()-t0:.1f}s)")
    return np.stack(X), np.array(y), np.array(folds)


def load_items_from_manifest(cfg, data_dir, limit=None):
    from src.datasets_esc50 import build_manifest
    from src.audio_io import load_wav, fix_length
    df = build_manifest(data_dir)
    if limit:
        df = df.groupby("group", group_keys=False).head(max(1, limit // df["group"].nunique()))
    items = []
    for _, r in df.iterrows():
        wav, sr = load_wav(r["filepath"])
        wav = fix_length(wav, sr, cfg["clip_seconds"])
        items.append({"wav": wav, "sr": sr,
                      "group_index": int(r["group_index"]), "fold": int(r["fold"])})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--self-test", action="store_true",
                    help="use synthetic separable audio (no download)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto",
                    help="auto|cuda|cpu (auto falls back to CPU if GPU mel is broken)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data_dir"]
    outdir = args.outdir or cfg["results_dir"]
    os.makedirs(outdir, exist_ok=True)

    device = resolve_feature_device(args.device)
    if args.device in ("auto", "cuda") and device == "cpu":
        print("[note] using CPU for mel features (GPU spectrogram unavailable here).")
    eg.validate()

    # --- data ---
    if args.self_test:
        from src.datasets_esc50 import make_synthetic_dataset
        items = make_synthetic_dataset(n_per_class=40, sr=cfg["sample_rate"],
                                       seconds=2.0, seed=cfg["seed"])
        source = "synthetic"
    else:
        meta_csv = os.path.join(data_dir, "meta", "esc50.csv")
        if not os.path.exists(meta_csv):
            raise FileNotFoundError(
                f"ESC-50 metadata not found at {meta_csv}. Run "
                "scripts/1_prepare_data.py or pass --self-test."
            )
        items = load_items_from_manifest(cfg, data_dir, limit=args.limit)
        source = "esc50"

    print(f"source={source}  clips={len(items)}  device={device}")

    # --- features + probe ---
    X, y, folds = build_features(items, cfg, device)
    print(f"features X={X.shape}  groups={len(np.unique(y))}")
    cv = P.crossval(X, y, folds, C=cfg["probe_C"], max_iter=cfg["probe_max_iter"],
                    seed=cfg["seed"])
    cls = M.classification_metrics(cv["y_true"], cv["y_pred"])
    chance = M.chance_level(y)

    print("\n=== RESULT: raw log-mel + linear probe ===")
    print(f"  mean accuracy : {cv['mean_accuracy']:.3f} +/- {cv['std_accuracy']:.3f}")
    print(f"  macro F1      : {cls['macro_f1']:.3f}")
    print(f"  chance (maj.) : {chance:.3f}")
    print(f"  per-fold acc  : {[round(a,3) for a in cv['fold_accuracy']]}")

    # --- save artifacts ---
    tag = f"mel_probe_{source}"
    np.savez(os.path.join(outdir, f"features_{tag}.npz"),
             X=X, y=y, folds=folds, group_names=np.array(eg.GROUP_NAMES))
    cm = M.confusion(cv["y_true"], cv["y_pred"], n_labels=len(eg.GROUP_NAMES))
    result = {
        "representation": "logmel_pool(mean+std)",
        "source": source,
        "feature_dim": int(X.shape[1]),
        "n_clips": int(X.shape[0]),
        "n_groups": int(len(eg.GROUP_NAMES)),
        "label": cfg["label"],
        "mean_accuracy": cv["mean_accuracy"],
        "std_accuracy": cv["std_accuracy"],
        "macro_f1": cls["macro_f1"],
        "chance_majority": chance,
        "fold_accuracy": cv["fold_accuracy"],
        "config": {k: cfg[k] for k in ("sample_rate", "n_fft", "hop_length", "n_mels")},
        # cost-parameter slots (filled by codec scripts; mel has no codec cost)
        "bitrate_kbps": None, "n_params": 0, "encode_latency_ms": None,
    }
    with open(os.path.join(outdir, f"result_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    fig = os.path.join(cfg["figures_dir"], f"confusion_{tag}.png")
    plot_confusion(cm, eg.GROUP_NAMES, fig, title=f"log-mel probe ({source})")
    print(f"\nsaved: result_{tag}.json, features_{tag}.npz, {os.path.basename(fig)}")


if __name__ == "__main__":
    main()
