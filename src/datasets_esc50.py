"""ESC-50 manifest + a synthetic fallback dataset for smoke tests.

The manifest reader uses pandas (present). The synthetic generator lets the
whole pipeline (features -> probe -> metrics -> figures) run end-to-end with
zero downloads, so you can verify correctness before committing to the
~600 MB ESC-50 download.
"""
from __future__ import annotations

import os
import numpy as np

from . import event_groups as eg


def build_manifest(data_dir: str):
    """Read meta/esc50.csv and attach designed event-group labels.

    Returns a pandas DataFrame with columns:
        filepath, filename, fold, category, target(0-49),
        group, group_index
    """
    import pandas as pd

    meta_csv = os.path.join(data_dir, "meta", "esc50.csv")
    audio_dir = os.path.join(data_dir, "audio")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(
            f"Could not find {meta_csv}. Run scripts/1_prepare_data.py first, "
            f"or pass --self-test to use synthetic audio."
        )
    df = pd.read_csv(meta_csv)
    df["filepath"] = df["filename"].apply(lambda fn: os.path.join(audio_dir, fn))
    df["group"] = df["category"].apply(eg.group_of)
    df["group_index"] = df["category"].apply(eg.group_index)
    return df


def make_synthetic_dataset(groups=None, n_per_class=40, sr=16000, seconds=2.0, seed=0):
    """Generate separable synthetic clips, one distinct timbre per group.

    Returns a list of dicts: {wav, sr, group, group_index, fold, category}.
    Each group gets a distinct fundamental + harmonics so a mel probe can
    separate them well above the majority-class baseline and verify the pipeline.
    """
    if groups is None:
        groups = eg.GROUP_NAMES
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    items = []
    for gi, g in enumerate(groups):
        base = 120.0 * (gi + 1)  # distinct fundamental per group
        for k in range(n_per_class):
            wav = np.zeros(n, dtype=np.float32)
            for h in range(1, 4):
                f = base * h * (1.0 + 0.02 * rng.standard_normal())
                wav += (1.0 / h) * np.sin(2 * np.pi * f * t).astype(np.float32)
            wav += 0.05 * rng.standard_normal(n).astype(np.float32)
            wav = (wav / (np.max(np.abs(wav)) + 1e-9)).astype(np.float32)
            items.append({
                "wav": wav, "sr": sr, "group": g, "group_index": gi,
                "fold": (k % 5) + 1, "category": f"synth_{g}",
            })
    rng.shuffle(items)
    return items
