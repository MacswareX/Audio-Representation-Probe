"""EnCodec discrete-token extraction (pretrained baseline).

GATED: requires `transformers` (+ `huggingface_hub`). The first time it runs it
downloads the EnCodec checkpoint (~90 MB for facebook/encodec_24khz).

Research instrument: build probe features from histograms of the first K
codebooks and measure how much designed event-group information is linearly
decodable from that interface.
"""
from __future__ import annotations

import numpy as np
import torch

from .features import resample_wave


def have_transformers() -> bool:
    import importlib.util
    return importlib.util.find_spec("transformers") is not None


def load_encodec(model_name="facebook/encodec_24khz", device="cpu"):
    from transformers import EncodecModel, AutoProcessor
    model = EncodecModel.from_pretrained(model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


@torch.no_grad()
def extract_codes(model, processor, wav, sr, bandwidth=6.0, device="cpu"):
    """Return discrete codes [n_q, T] (int) for one clip."""
    target_sr = processor.sampling_rate
    x = resample_wave(wav, sr, target_sr, device="cpu").numpy()
    inputs = processor(raw_audio=x, sampling_rate=target_sr, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    mask = inputs.get("padding_mask")
    if mask is not None:
        mask = mask.to(device)
    enc = model.encode(input_values, mask, bandwidth=bandwidth)
    codes = enc.audio_codes            # [n_chunks, B, n_q, T]
    codes = codes[0, 0]                # -> [n_q, T]
    return codes.detach().cpu().numpy().astype(np.int64)


def codes_to_histogram_feature(codes, codebook_size=1024, n_codebooks=2):
    """codes [n_q, T] -> normalized bag-of-codes over the first n_codebooks.

    Feature dim = n_codebooks * codebook_size. The first-k prefix is a chosen
    experimental interface; superiority of early codebooks is not assumed.
    """
    nq = min(n_codebooks, codes.shape[0])
    feats = []
    for q in range(nq):
        h = np.bincount(codes[q], minlength=codebook_size).astype(np.float32)
        s = h.sum()
        feats.append(h / s if s > 0 else h)
    return np.concatenate(feats, axis=0)
