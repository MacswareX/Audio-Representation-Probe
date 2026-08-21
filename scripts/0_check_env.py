"""Environment check. Prints what is ready and what each script needs.

Run:  python scripts/0_check_env.py
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def have(pkg):
    return importlib.util.find_spec(pkg) is not None


def line(name, ok, extra=""):
    print(f"  [{'OK ' if ok else 'MISS'}] {name:16s} {extra}")


def main():
    print("=== core (needed for the zero-dependency mel baseline) ===")
    core = ["torch", "torchaudio", "numpy", "scipy", "sklearn", "pandas",
            "matplotlib", "yaml"]
    core_ok = True
    for p in core:
        ok = have(p)
        core_ok = core_ok and ok
        line(p, ok)

    print("\n=== optional (needed for the EnCodec / CLAP token paths) ===")
    optional = ["transformers", "huggingface_hub", "soundfile", "datasets", "snac"]
    for p in optional:
        line(p, have(p))

    print("\n=== torch / device ===")
    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  torch {torch.__version__}  cuda_available={cuda_ok}")
        if cuda_ok:
            p = torch.cuda.get_device_properties(0)
            print(f"  gpu: {p.name}  vram={p.total_memory/1024**3:.1f} GB")
    except Exception as e:
        print("  torch error:", e)

    print("\n=== torchaudio I/O backend ===")
    try:
        import torchaudio
        backends = torchaudio.list_audio_backends()
        print(f"  backends={backends}  (empty is OK: we load WAV via scipy)")
    except Exception as e:
        print("  torchaudio error:", e)

    print("\n=== Phase-2 audio I/O (non-WAV: FSD50K extras / AudioSet mp3) ===")
    try:
        from src.audio_io import get_ffmpeg_exe
        exe = get_ffmpeg_exe()
        if exe:
            print(f"  ffmpeg: {exe}")
        else:
            print("  ffmpeg: NOT found -> `pip install imageio-ffmpeg` or add to PATH")
    except Exception as e:
        print("  ffmpeg probe error:", e)
    for p in ["imageio_ffmpeg", "librosa"]:
        line(p, have(p))

    print("\n=== GPU mel spectrogram (nvrtc JIT) sanity ===")
    mel_cuda_ok = False
    try:
        from src.features import resolve_feature_device
        dev = resolve_feature_device("cuda")
        if dev == "cuda":
            mel_cuda_ok = True
            print("  GPU mel works.")
        else:
            print("  GPU mel unavailable here -> features auto-fall back to CPU (fine).")
    except Exception as e:
        print("  mel probe error:", e)

    print("\n=== taxonomy ===")
    try:
        from src import event_groups as eg
        eg.validate()
        print(f"  OK: 50 ESC-50 categories -> {len(eg.GROUP_NAMES)} designed event groups")
    except Exception as e:
        print("  taxonomy error:", e)

    print("\n=== verdict ===")
    encodec_ready = have("transformers") and have("huggingface_hub")
    if core_ok:
        print("  Core release pipeline is ready.")
        if cuda_ok and mel_cuda_ok:
            print("  CUDA acceleration is ready:")
            device = "cuda"
        else:
            print("  CUDA unavailable; feature extraction will use CPU.")
            device = "cpu"
        print(f"    python scripts/2_mel_probe_baseline.py --device {device}")
        print(f"    python scripts/6_train_mel_vqvae.py --device {device}")
    else:
        print("  NOT release-ready: install all core dependencies.")
    if encodec_ready:
        print("  EnCodec/Transformers dependencies are ready.")
    else:
        print("  EnCodec baseline needs: pip install -r requirements-extra.txt")
    if not have("snac"):
        print("  SNAC is optional and does not block the current benchmark.")


if __name__ == "__main__":
    main()
