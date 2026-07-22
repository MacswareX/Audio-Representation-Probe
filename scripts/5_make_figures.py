"""Make summary figures from result JSON files.

    python scripts/5_make_figures.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.visualize import plot_probe_performance, plot_relative_probe_accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--source", default="esc50")
    args = ap.parse_args()
    cfg = load_config(args.config)
    results_dir = cfg["results_dir"]
    figures_dir = cfg["figures_dir"]

    summary = os.path.join(results_dir, "event_retention_summary.json")
    if not os.path.exists(summary):
        print("Summary not found. Run scripts/4_eval_event_retention.py first.")
        sys.exit(2)

    with open(summary, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("source") != args.source:
        print(f"Summary source is {data.get('source')!r}, expected {args.source!r}. "
              "Re-run script 4 with the desired --source.")
        sys.exit(2)

    rows = data["rows"]
    labels = [r["representation"] for r in rows]
    accs = [r["mean_accuracy"] for r in rows]
    f1s = [r["macro_f1"] for r in rows]
    relative = [r["relative_probe_accuracy_vs_logmel"] for r in rows]
    chance = rows[0].get("chance_majority") if rows else None
    if not labels:
        print("Summary contains no rows. Run script 4 first.")
        sys.exit(2)

    os.makedirs(figures_dir, exist_ok=True)
    performance = os.path.join(figures_dir, f"probe_performance_{args.source}.png")
    ratio = os.path.join(figures_dir, f"relative_probe_accuracy_{args.source}.png")
    plot_probe_performance(labels, accs, f1s, performance, chance=chance)
    plot_relative_probe_accuracy(labels, relative, ratio)
    print("saved", performance)
    print("saved", ratio)


if __name__ == "__main__":
    main()
