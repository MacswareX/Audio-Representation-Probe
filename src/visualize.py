"""Plotting helpers (matplotlib, Agg backend so it works headless)."""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_confusion(cm, labels, path, title="Confusion matrix"):
    _ensure_dir(path)
    cm = np.asarray(cm, dtype=np.float64)
    row = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)
    fig, ax = plt.subplots(figsize=(1.2 + 0.6 * len(labels), 1.0 + 0.6 * len(labels)))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_probe_performance(labels, accuracies, macro_f1, path, chance=None,
                           title="Held-out embodied-group probe performance"):
    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(2.0 + 1.5 * len(labels), 4.5))
    x = np.arange(len(labels))
    width = 0.36
    bars_acc = ax.bar(x - width / 2, accuracies, width, label="accuracy", color="#3b6ea5")
    bars_f1 = ax.bar(x + width / 2, macro_f1, width, label="macro-F1", color="#bc6c45")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("held-out score")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    if chance is not None:
        ax.axhline(chance, ls="--", c="grey", label=f"majority baseline = {chance:.2f}")
    ax.legend()
    for bars in (bars_acc, bars_f1):
        for b in bars:
            value = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, value + 0.012, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_relative_probe_accuracy(labels, relative, path):
    """Secondary ratio versus log-mel; values above one are valid."""
    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(2.0 + 1.4 * len(labels), 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, relative, color="#4f7f6b")
    ax.axhline(1.0, ls="--", c="grey", label="pooled log-mel reference = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("relative probe accuracy")
    ax.set_ylim(0, max(1.25, max(relative) + 0.15))
    ax.set_title("Relative probe accuracy versus pooled log-mel")
    ax.legend()
    for b, value in zip(bars, relative):
        ax.text(b.get_x() + b.get_width() / 2, value + 0.02, f"{value:.2f}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
