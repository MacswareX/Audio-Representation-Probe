"""Download + unpack ESC-50, then build the embodied-group manifest.

Run:  python scripts/1_prepare_data.py            # downloads to ./data
      python scripts/1_prepare_data.py --check    # just validate an existing copy

ESC-50: 2000 clips, 5 s, 44.1 kHz mono WAV, 50 classes, CC BY-NC.
Source: https://github.com/karolpiczak/ESC-50  (~600 MB master.zip)
"""
import argparse
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config

ZIP_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"


def download(url, dest):
    import requests
    print(f"Downloading {url}\n  -> {dest}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done/1e6:6.1f}/{total/1e6:6.1f} MB ({pct:4.1f}%)",
                          end="", flush=True)
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--check", action="store_true",
                    help="only validate an existing copy, do not download")
    args = ap.parse_args()
    cfg = load_config(args.config)

    os.makedirs(args.data_root, exist_ok=True)
    data_dir = os.path.join(args.data_root, "ESC-50-master")
    meta_csv = os.path.join(data_dir, "meta", "esc50.csv")

    if not args.check and not os.path.exists(meta_csv):
        zip_path = os.path.join(args.data_root, "ESC-50-master.zip")
        if not os.path.exists(zip_path):
            download(ZIP_URL, zip_path)
        print("Extracting...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(args.data_root)

    if not os.path.exists(meta_csv):
        print("ESC-50 not found. Re-run without --check to download, or place the "
              "dataset at", data_dir)
        sys.exit(2)

    from src.datasets_esc50 import build_manifest
    df = build_manifest(data_dir)
    print(f"manifest: {len(df)} clips, {df['category'].nunique()} categories")
    print("clips per embodied group:")
    print(df["group"].value_counts().to_string())
    out = os.path.join(args.data_root, "manifest.csv")
    df.to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
