"""Configuration loading.

Plain-dict config with sane defaults, optionally overlaid by a YAML file.
Kept dependency-light on purpose (PyYAML is the only extra, and it is already
present in the environment).
"""
from __future__ import annotations

import copy
import os

try:
    import yaml  # PyYAML, present in the base environment
except Exception:  # pragma: no cover
    yaml = None

DEFAULTS = {
    # --- audio / features ---
    "sample_rate": 16000,        # probe/mel sample rate (resampled from source)
    "n_fft": 1024,
    "hop_length": 320,           # 50 frames/sec at 16 kHz
    "n_mels": 64,
    "f_min": 20,
    "f_max": 8000,
    "clip_seconds": 5.0,         # ESC-50 clips are 5 s; pad/trim to this
    # --- labels ---
    "label": "group",            # "group" (embodied event groups) or "category" (50 ESC classes)
    # --- probe ---
    "probe_C": 1.0,
    "probe_max_iter": 2000,
    "seed": 0,
    # --- EnCodec (gated; requires transformers) ---
    "encodec_model": "facebook/encodec_24khz",
    "encodec_bandwidth": 6.0,    # kbps -> 8 codebooks at 24 kHz
    "encodec_sample_rate": 24000,
    "encodec_probe_codebooks": 2,  # use the FIRST K codebooks to probe coarse-token semantics
    # --- custom mel-RVQ tokenizer (scripts/6) ---
    "mel_frames": 256,             # fixed mel time frames fed to the tokenizer
    "melrvq_latent_dim": 64,
    "melrvq_codebook_size": 256,
    "melrvq_num_quantizers": 4,
    "melrvq_probe_codebooks": 2,   # probe the FIRST K learned codebooks
    "melrvq_epochs": 20,
    "melrvq_batch_size": 32,
    "melrvq_lr": 0.0003,
    # --- paths ---
    "data_dir": "data/ESC-50-master",
    "results_dir": "assets/results",
    "figures_dir": "assets/figures",
}


def load_config(path: str | None = None) -> dict:
    """Return DEFAULTS overlaid with values from a YAML file (if given)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path and os.path.exists(path) and yaml is not None:
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in user.items() if v is not None})
    return cfg
