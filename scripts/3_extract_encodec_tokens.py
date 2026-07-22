"""Extract EnCodec discrete tokens and build coarse-codebook probe features.

GATED: needs `transformers` + `huggingface_hub` (see requirements-extra.txt).
First run downloads facebook/encodec_24khz (~90 MB).

    pip install -r requirements-extra.txt
    python scripts/3_extract_encodec_tokens.py --self-test      # quick check
    python scripts/3_extract_encodec_tokens.py                  # real ESC-50

Produces features_encodec_<source>.npz consumed by 4_eval_event_retention.py.
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
from src import metrics as M
from src.encodec_tokens import (have_transformers, load_encodec, extract_codes,
                                 codes_to_histogram_feature)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if not have_transformers():
        print("transformers not installed. Run:\n"
              "  pip install -r requirements-extra.txt")
        sys.exit(2)

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data_dir"]
    outdir = args.outdir or cfg["results_dir"]
    os.makedirs(outdir, exist_ok=True)
    eg.validate()

    from src.features import resolve_compute_device
    device = resolve_compute_device(args.device)

    # data
    if args.self_test:
        from src.datasets_esc50 import make_synthetic_dataset
        items = make_synthetic_dataset(n_per_class=20, sr=cfg["sample_rate"],
                                       seconds=2.0, seed=cfg["seed"])
        source = "synthetic"
    else:
        meta_csv = os.path.join(data_dir, "meta", "esc50.csv")
        if not os.path.exists(meta_csv):
            raise FileNotFoundError(
                f"ESC-50 metadata not found at {meta_csv}. Run "
                "scripts/1_prepare_data.py or pass --self-test."
            )
        from src.datasets_esc50 import build_manifest
        from src.audio_io import load_wav, fix_length
        df = build_manifest(data_dir)
        if args.limit:
            df = df.groupby("group", group_keys=False).head(
                max(1, args.limit // df["group"].nunique()))
        items = []
        for _, r in df.iterrows():
            wav, sr = load_wav(r["filepath"])
            wav = fix_length(wav, sr, cfg["clip_seconds"])
            items.append({"wav": wav, "sr": sr,
                          "group_index": int(r["group_index"]), "fold": int(r["fold"])})
        source = "esc50"

    print(f"source={source} clips={len(items)} device={device} "
          f"bandwidth={cfg['encodec_bandwidth']}kbps "
          f"probe_codebooks={cfg['encodec_probe_codebooks']}")

    model, processor = load_encodec(cfg["encodec_model"], device=device)
    n_params = sum(p.numel() for p in model.parameters())
    codebook_size = int(model.config.codebook_size)

    # Warmup + device fallback (mirrors the mel CPU fallback in env_01).
    try:
        _ = extract_codes(model, processor, items[0]["wav"], items[0]["sr"],
                          bandwidth=cfg["encodec_bandwidth"], device=device)
    except RuntimeError as e:
        print(f"[note] GPU encode failed ({str(e)[:80]}...); falling back to CPU.")
        device = "cpu"
        model, processor = load_encodec(cfg["encodec_model"], device=device)

    X, y, folds = [], [], []
    all_codes_cb0 = []
    t0 = time.time()
    for i, it in enumerate(items):
        codes = extract_codes(model, processor, it["wav"], it["sr"],
                              bandwidth=cfg["encodec_bandwidth"], device=device)
        feat = codes_to_histogram_feature(codes, codebook_size=codebook_size,
                                          n_codebooks=cfg["encodec_probe_codebooks"])
        X.append(feat)
        y.append(it["group_index"])
        folds.append(it["fold"])
        all_codes_cb0.append(codes[0])
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.1f}s)")
    enc_latency_ms = 1000.0 * (time.time() - t0) / max(len(items), 1)

    X = np.stack(X); y = np.array(y); folds = np.array(folds)
    util = M.codebook_utilization(np.concatenate(all_codes_cb0), codebook_size)

    tag = f"encodec_{source}"
    np.savez(os.path.join(outdir, f"features_{tag}.npz"),
             X=X, y=y, folds=folds, group_names=np.array(eg.GROUP_NAMES))
    meta = {
        "representation": f"encodec_codehist(first{cfg['encodec_probe_codebooks']}cb)",
        "source": source, "feature_dim": int(X.shape[1]), "n_clips": int(X.shape[0]),
        "encodec_model": cfg["encodec_model"], "bandwidth_kbps": cfg["encodec_bandwidth"],
        "codebook_size": codebook_size, "codebook0_utilization": util,
        "n_params": int(n_params), "encode_latency_ms": enc_latency_ms,
    }
    with open(os.path.join(outdir, f"meta_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved features_{tag}.npz ; codebook0 util={util['utilization']:.2f} "
          f"perplexity={util['perplexity']:.1f} ; latency={enc_latency_ms:.1f} ms/clip")
    print("next: python scripts/4_eval_event_retention.py")


if __name__ == "__main__":
    main()
