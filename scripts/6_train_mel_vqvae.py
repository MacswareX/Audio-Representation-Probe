"""Train the small mel-RVQ tokenizer and export its token features.

This custom tokenizer path is compared with the pretrained EnCodec interface. It:
  1. builds a fixed-size log-mel dataset (synthetic or ESC-50),
  2. trains MelRVQVAE (reconstruction + VQ loss) with EMA codebooks,
  3. tracks codebook utilization/perplexity,
  4. exports per-clip token features (bag-of-codes over the first K codebooks)
     to features_melrvq_<source>.npz for scripts/4_eval_event_retention.py,
  5. records cost parameters (bitrate, params, latency) in a result JSON.

Mel features run on CPU when the GPU spectrogram path is unavailable (env_01);
the neural net trains on CUDA when possible.

    python scripts/6_train_mel_vqvae.py --self-test    # synthetic, full pipeline
    python scripts/6_train_mel_vqvae.py --smoke        # synthetic, 5 epochs (fast)
    python scripts/6_train_mel_vqvae.py                # ESC-50
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
from src.features import LogMel, resolve_feature_device, resolve_compute_device
from src import metrics as M


def load_items(cfg, data_dir, self_test, limit):
    if self_test:
        from src.datasets_esc50 import make_synthetic_dataset
        items = make_synthetic_dataset(n_per_class=40, sr=cfg["sample_rate"],
                                       seconds=2.0, seed=cfg["seed"])
        return items, "synthetic"
    meta_csv = os.path.join(data_dir, "meta", "esc50.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(
            f"ESC-50 metadata not found at {meta_csv}. Run "
            "scripts/1_prepare_data.py or pass --self-test."
        )
    from src.datasets_esc50 import build_manifest
    from src.audio_io import load_audio, fix_length
    df = build_manifest(data_dir)
    if limit:
        df = df.groupby("group", group_keys=False).head(
            max(1, limit // df["group"].nunique()))
    items = []
    for _, r in df.iterrows():
        wav, sr = load_audio(r["filepath"], target_sr=cfg["sample_rate"])
        wav = fix_length(wav, sr, cfg["clip_seconds"])
        items.append({"wav": wav, "sr": sr,
                      "group_index": int(r["group_index"]), "fold": int(r["fold"])})
    return items, "esc50"


def build_mel_dataset(items, cfg, feat_device):
    import torch
    lm = LogMel(sample_rate=cfg["sample_rate"], n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
                f_min=cfg["f_min"], f_max=cfg["f_max"], device=feat_device)
    T = cfg["mel_frames"]
    mels, y, folds = [], [], []
    for it in items:
        m = lm.wave_to_logmel(it["wav"], it["sr"]).detach().cpu() / 80.0  # ~[-1,0]
        if m.shape[1] < T:
            m = torch.nn.functional.pad(m, (0, T - m.shape[1]))
        else:
            m = m[:, :T]
        mels.append(m)
        y.append(it["group_index"])
        folds.append(it["fold"])
    X = torch.stack(mels).unsqueeze(1)  # [N, 1, n_mels, T]
    return X, np.array(y), np.array(folds)


def train(model, X, epochs, batch_size, lr, device, cb_size, verbose=True):
    import torch
    import torch.nn.functional as F
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    history = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = recon_t = vq_t = 0.0
        last_cb0 = None
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            mb = X[idx].to(device)
            target = mb.mean(dim=3).squeeze(1)          # [B, n_mels]
            out = model(mb)
            recon = F.mse_loss(out["recon"], target)
            loss = recon + out["vq_loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            bs = mb.shape[0]
            tot += float(loss) * bs
            recon_t += float(recon) * bs
            vq_t += float(out["vq_loss"]) * bs
            last_cb0 = out["indices"][0].detach().cpu().numpy()
        util = M.codebook_utilization(last_cb0, cb_size)
        history.append({"epoch": ep, "loss": tot / n, "recon": recon_t / n,
                        "vq": vq_t / n, "cb0_util": util["utilization"],
                        "cb0_perplexity": util["perplexity"]})
        if verbose:
            print(f"  epoch {ep:2d}: loss={tot/n:.4f} recon={recon_t/n:.4f} "
                  f"vq={vq_t/n:.4f} cb0_util={util['utilization']:.2f} "
                  f"ppl={util['perplexity']:.1f}")
    return history


def extract_token_features(model, X, K, cb_size, device):
    import torch
    model.to(device).eval()
    feats, cb0 = [], []
    latent_T = None
    with torch.no_grad():
        for i in range(X.shape[0]):
            out = model(X[i:i + 1].to(device))
            idxs = out["indices"]
            latent_T = idxs[0].shape[1]
            parts = []
            for q in range(min(K, len(idxs))):
                ids = idxs[q].reshape(-1).cpu().numpy()
                h = np.bincount(ids, minlength=cb_size).astype(np.float32)
                s = h.sum()
                parts.append(h / s if s > 0 else h)
            feats.append(np.concatenate(parts))
            cb0.append(idxs[0].reshape(-1).cpu().numpy())
    return np.stack(feats), np.concatenate(cb0), latent_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="synthetic, 5 epochs (fast)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    import torch
    from src.vqvae import MelRVQVAE

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data_dir"]
    outdir = args.outdir or cfg["results_dir"]
    os.makedirs(outdir, exist_ok=True)
    eg.validate()

    self_test = args.self_test or args.smoke
    epochs = args.epochs or (5 if args.smoke else cfg["melrvq_epochs"])
    feat_device = resolve_feature_device(args.device)
    comp_device = resolve_compute_device(args.device)
    if args.device in ("auto", "cuda") and feat_device == "cpu":
        print("[note] mel features on CPU (GPU spectrogram unavailable here).")
    print(f"feature_device={feat_device}  compute_device={comp_device}  epochs={epochs}")

    items, source = load_items(cfg, data_dir, self_test, args.limit)
    print(f"source={source}  clips={len(items)}")

    X, y, folds = build_mel_dataset(items, cfg, feat_device)
    print(f"mel dataset: {tuple(X.shape)}")

    cb = cfg["melrvq_codebook_size"]
    nq = cfg["melrvq_num_quantizers"]
    model = MelRVQVAE(n_mels=cfg["n_mels"], latent_dim=cfg["melrvq_latent_dim"],
                      num_quantizers=nq, codebook_size=cb)
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    history = train(model, X, epochs, cfg["melrvq_batch_size"], cfg["melrvq_lr"],
                    comp_device, cb)
    train_time = time.time() - t0

    K = cfg["melrvq_probe_codebooks"]
    t1 = time.time()
    feats, cb0_all, latent_T = extract_token_features(model, X, K, cb, comp_device)
    enc_latency_ms = 1000.0 * (time.time() - t1) / max(X.shape[0], 1)

    # cost parameters
    frame_rate = latent_T / cfg["clip_seconds"]                 # tokens/sec per codebook
    bits_per_frame = nq * np.log2(cb)
    bitrate_kbps = frame_rate * bits_per_frame / 1000.0
    util = M.codebook_utilization(cb0_all, cb)

    tag = f"melrvq_{source}"
    np.savez(os.path.join(outdir, f"features_{tag}.npz"),
             X=feats, y=y, folds=folds, group_names=np.array(eg.GROUP_NAMES))
    result = {
        "representation": f"melrvq_codehist(first{K}cb)",
        "source": source, "feature_dim": int(feats.shape[1]),
        "n_clips": int(feats.shape[0]),
        "num_quantizers": nq, "codebook_size": cb,
        "latent_frames": int(latent_T), "token_rate_hz_per_cb": float(frame_rate),
        "bitrate_kbps": float(bitrate_kbps),
        "codebook0": util,
        "n_params": int(n_params), "train_seconds": train_time,
        "encode_latency_ms": enc_latency_ms, "epochs": epochs,
        "history": history,
    }
    with open(os.path.join(outdir, f"result_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== mel-RVQ tokenizer ({source}) ===")
    print(f"  params={n_params/1e6:.2f}M  bitrate~{bitrate_kbps:.2f} kbps "
          f"({nq} codebooks x {cb} @ {frame_rate:.1f} Hz)")
    print(f"  codebook0 util={util['utilization']:.2f} perplexity={util['perplexity']:.1f}")
    print(f"  latency={enc_latency_ms:.1f} ms/clip  train={train_time:.1f}s")
    print(f"  saved features_{tag}.npz + result_{tag}.json")
    print("  next: python scripts/4_eval_event_retention.py")


if __name__ == "__main__":
    main()
