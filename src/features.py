"""Log-mel features via torchaudio *transforms* (pure tensor ops, no I/O backend).

torchaudio.transforms.MelSpectrogram / AmplitudeToDB are nn.Modules that run
on CPU or GPU and do NOT require an audio I/O backend, so they work even when
`torchaudio.list_audio_backends()` is empty.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio


def resample_wave(wav, orig_sr: int, target_sr: int, device: str = "cpu") -> torch.Tensor:
    """Resample a 1-D waveform (np array or tensor) to target_sr. Returns 1-D tensor."""
    x = torch.as_tensor(np.asarray(wav), dtype=torch.float32, device=device)
    if x.ndim > 1:
        x = x.mean(dim=0)
    if orig_sr != target_sr:
        x = torchaudio.functional.resample(x, orig_sr, target_sr)
    return x


class LogMel:
    """Compute log-mel spectrograms. Reusable across baselines."""

    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=320,
                 n_mels=64, f_min=20, f_max=8000, device="cpu"):
        self.sample_rate = sample_rate
        self.device = device
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max or sample_rate // 2,
            power=2.0,
        ).to(device)
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(device)

    def wave_to_logmel(self, wav, sr: int) -> torch.Tensor:
        """1-D waveform -> log-mel [n_mels, T]."""
        x = resample_wave(wav, sr, self.sample_rate, device=self.device)
        if x.numel() < self.melspec.n_fft:
            pad = self.melspec.n_fft - x.numel()
            x = torch.nn.functional.pad(x, (0, pad))
        m = self.melspec(x.unsqueeze(0))   # [1, n_mels, T]
        m = self.to_db(m)
        return m.squeeze(0)                # [n_mels, T]


def pool_stats(logmel: torch.Tensor) -> np.ndarray:
    """[n_mels, T] -> [2*n_mels] (mean and std over time). Time-invariant probe input."""
    mean = logmel.mean(dim=1)
    std = logmel.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=0).detach().cpu().numpy().astype(np.float32)


_DEVICE_CACHE = {}


def resolve_feature_device(requested: str = "auto") -> str:
    """Return a device on which torchaudio's MelSpectrogram actually works.

    On some CUDA installs (e.g. env_01 here, which is missing
    nvrtc-builtins64_124.dll), the complex-abs step inside torchaudio's
    spectrogram JIT-compiles a CUDA kernel and crashes. Mel is cheap, so we
    probe once and transparently fall back to CPU. Neural nets (conv/linear)
    are unaffected and can still use CUDA.
    """
    if requested in _DEVICE_CACHE:
        return _DEVICE_CACHE[requested]
    want_cuda = requested in ("auto", "cuda") and torch.cuda.is_available()
    dev = "cpu"
    if want_cuda:
        try:
            lm = LogMel(device="cuda")
            _ = lm.wave_to_logmel(np.zeros(2048, dtype=np.float32), 16000)
            dev = "cuda"
        except Exception:
            dev = "cpu"
    _DEVICE_CACHE[requested] = dev
    return dev


def resolve_compute_device(requested: str = "auto") -> str:
    """Device for neural nets (conv/linear). Uses CUDA if available; these ops
    do not depend on the nvrtc JIT path that breaks mel on some installs."""
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
