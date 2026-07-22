"""Audio loading.

Two layers:
  1. `load_wav`  -- WAV via `scipy.io.wavfile`, zero extra dependencies. This is
     all Phase-1 (ESC-50) and FSD50K (also WAV) need, and it works even when
     `torchaudio.list_audio_backends()` is empty.
  2. `load_audio` -- general loader (mp3/flac/ogg/m4a/...) via an ffmpeg binary,
     discovered from imageio-ffmpeg, a known install path, or PATH. This unlocks
     Phase-2/3 datasets such as AudioSet.

Design stays dependency-light: WAV never shells out; only non-WAV uses ffmpeg.
"""
from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess

import numpy as np
from scipy.io import wavfile

def to_float32(data: np.ndarray) -> np.ndarray:
    """Convert common PCM dtypes to float32 in [-1, 1]."""
    if data.dtype == np.int16:
        return data.astype(np.float32) / 32768.0
    if data.dtype == np.int32:
        return data.astype(np.float32) / 2147483648.0
    if data.dtype == np.uint8:  # WAV uint8 is unsigned, centered at 128
        return (data.astype(np.float32) - 128.0) / 128.0
    if data.dtype in (np.float32, np.float64):
        return data.astype(np.float32)
    return data.astype(np.float32)


def load_wav(path: str, mono: bool = True):
    """Load a WAV file. Returns (waveform float32 [T], sample_rate int)."""
    sr, data = wavfile.read(path)
    data = to_float32(data)
    if mono and data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def fix_length(wav: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """Pad with zeros or trim so the clip is exactly `seconds` long."""
    target = int(round(sr * seconds))
    if len(wav) >= target:
        return wav[:target]
    out = np.zeros(target, dtype=np.float32)
    out[: len(wav)] = wav
    return out


@functools.lru_cache(maxsize=1)
def get_ffmpeg_exe():
    """Locate ffmpeg. Order: AET_FFMPEG -> imageio-ffmpeg -> PATH."""
    configured = os.environ.get("AET_FFMPEG")
    if configured and os.path.isfile(configured):
        return configured
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass
    return shutil.which("ffmpeg")


def have_ffmpeg() -> bool:
    return get_ffmpeg_exe() is not None


def _probe_sr(path: str):
    """Best-effort native sample-rate probe by parsing `ffmpeg -i` stderr."""
    exe = get_ffmpeg_exe()
    if exe is None:
        return None
    p = subprocess.run([exe, "-hide_banner", "-i", path],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    txt = (p.stderr or b"").decode("utf-8", "ignore")
    m = re.search(r"(\d+) Hz", txt)
    return int(m.group(1)) if m else None


def _ffmpeg_decode(path: str, target_sr: int, mono: bool = True) -> np.ndarray:
    """Decode any audio file to float32 PCM at target_sr via ffmpeg (pipe)."""
    exe = get_ffmpeg_exe()
    if exe is None:
        raise RuntimeError(
            "ffmpeg not found. `pip install imageio-ffmpeg` or add ffmpeg to PATH.")
    cmd = [exe, "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
           "-ar", str(target_sr)]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def load_audio(path: str, target_sr: int | None = None, mono: bool = True):
    """General loader. WAV -> scipy (native sr). Other formats -> ffmpeg.

    For non-WAV, audio is decoded at `target_sr` (or the probed native rate,
    or 16000 as a last resort). Returns (waveform float32 [T], sample_rate).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return load_wav(path, mono=mono)
    sr = target_sr or _probe_sr(path) or 16000
    return _ffmpeg_decode(path, target_sr=sr, mono=mono), int(sr)
