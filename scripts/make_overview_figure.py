#!/usr/bin/env python
"""Polished, data-driven system overview figure for the README.

Four stages, each with a real mini-visualization:
  1. Vibration under varying speed (synthetic waveform + RPM).
  2. Order tracking -> order spectrum (real ``order_spectrum``, labeled fault lines).
  3. Time-gap-aware Transformer (real order-domain feature tokens + attention).
  4. Causal EOL smoothing (real honest Train1 RUL trajectory).

Writes ``figures/results/overview.png`` / ``.pdf``.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phmkorea_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

from phm_pipeline.features import order_spectrum

OUT = ROOT / "figures" / "results"
FEATURES = ROOT / "artifacts/features/train_full_order_domain.parquet"
OOF = ROOT / "artifacts/runs/lobo_order_domain_nested/oof_predictions_honest.csv"
OOF_RAW = ROOT / "artifacts/runs/lobo_order_domain_nested/oof_predictions_raw.csv"

INK = "#101828"
SLATE = "#475467"
GRID = "#E4E7EC"
TEAL = "#0E9384"
CORAL = "#F97066"
AMBER = "#F79009"
NAVY = "#1D4E89"
CARD_BG = "#FCFDFE"
TOKEN_CMAP = LinearSegmentedColormap.from_list("teal", ["#EAF6F4", "#5BC3B4", "#0E9384", "#0A5852"])

STAGES = [
    ("1", "Vibration + speed", "60 s @ 25.6 kHz,\nvarying shaft RPM", NAVY),
    ("2", "Order tracking", "resample time -> shaft angle;\nfault lines fixed in order", TEAL),
    ("3", "Time-gap-aware Transformer", "causal window of order\nfeatures + elapsed-time gaps", AMBER),
    ("4", "Causal EOL smoothing", "running quantile of the\npredicted end-of-life", CORAL),
]

# Layout in 0..100 x 0..40 canvas units.
CARD_W, CARD_H, CARD_Y = 22.0, 25.0, 6.0
CARD_X = [1.5, 26.0, 50.5, 75.0]


def _ax_frac(canvas, x, y, w, h):
    fig = canvas.figure
    p0 = fig.transFigure.inverted().transform(canvas.transData.transform((x, y)))
    p1 = fig.transFigure.inverted().transform(canvas.transData.transform((x + w, y + h)))
    return [p0[0], p0[1], p1[0] - p0[0], p1[1] - p0[1]]


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def mini_vibration(ax):
    t = np.linspace(0, 1, 700)
    env = 0.35 + 0.75 * t
    rng = np.random.default_rng(3)
    vib = env * np.sin(2 * np.pi * 21 * t) + 0.12 * env * rng.standard_normal(t.size)
    ax.plot(t, vib, color=NAVY, lw=0.8)
    ax.plot(t, 1.55 + 0.55 * np.tanh(4 * (t - 0.3)), color=AMBER, lw=2.0)  # RPM
    ax.text(0.02, 1.9, "RPM", color=AMBER, fontsize=7.5, fontweight="bold")
    ax.set_ylim(-2.1, 2.3)
    _clean(ax)


def mini_order(ax):
    fs, n = 25_600.0, int(25_600 * 1.4)
    t = np.arange(n) / fs
    rpm = np.linspace(1500, 2300, n)
    phase = 2 * np.pi * np.cumsum(rpm / 60.0) / fs
    rng = np.random.default_rng(1)
    sig = np.sin(phase) + 0.7 * np.sin(3.58 * phase) + 0.4 * np.sin(5.42 * phase) + 0.12 * rng.standard_normal(n)
    orders, amp = order_spectrum(sig, rpm, sample_rate_hz=fs, samples_per_revolution=128, max_order=8.0)
    amp = amp / (amp.max() + 1e-9)
    ax.plot(orders, amp, color=TEAL, lw=1.4)
    ax.fill_between(orders, amp, color=TEAL, alpha=0.18)
    for o, name in [(3.58, "BPFO"), (5.42, "BPFI")]:
        ax.axvline(o, color=CORAL, lw=0.9, ls=":")
        ax.text(o, 1.04, name, ha="center", color=CORAL, fontsize=7)
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 8)
    _clean(ax)


def mini_transformer(ax):
    df = pd.read_parquet(FEATURES)
    g = df[df["run_id"] == "Train1"].sort_values("mid_time_s")
    cols = [
        "ch3_order_bpfo_energy_sum", "ch3_order_bpfi_energy_sum", "ch3_order_bsf_energy_sum",
        "ch3_order_ftf_energy_sum", "ch3_env_order_bpfi_energy_sum", "ch3_env_order_ftf_energy_sum",
    ]
    m = g[cols].to_numpy(dtype=float)
    m = (m - np.nanmin(m, 0)) / (np.nanmax(m, 0) - np.nanmin(m, 0) + 1e-9)
    idx = np.linspace(0, len(g) - 1, 7).astype(int)
    tokens = m[idx]  # (7, 6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _clean(ax)
    n = len(tokens)
    gaps = np.linspace(0.9, 1.5, n - 1)
    xpos = np.concatenate([[0.0], np.cumsum(gaps)])
    xpos = 0.05 + 0.9 * xpos / xpos[-1]
    tw, y0, th = 0.055, 0.08, 0.42
    centers = []
    for xi, tok in zip(xpos, tokens):
        for k, val in enumerate(tok):
            ax.add_patch(plt.Rectangle((xi, y0 + k * th / len(tok)), tw, th / len(tok),
                                       facecolor=TOKEN_CMAP(val), edgecolor="white", lw=0.4))
        centers.append(xi + tw / 2)
    # causal attention arcs into the last token
    for c in centers[:-1]:
        ax.add_patch(FancyArrowPatch((c, y0 + th + 0.03), (centers[-1], y0 + th + 0.03),
                                     connectionstyle="arc3,rad=0.55", arrowstyle="-",
                                     color=AMBER, lw=0.8, alpha=0.7))
    ax.text(0.5, 0.98, "self-attention", ha="center", va="top", color=AMBER, fontsize=7.5, fontweight="bold")
    ax.annotate("", xy=(centers[-1] + 0.02, y0 + th / 2), xytext=(centers[-1] + 0.14, y0 + th / 2),
                arrowprops={"arrowstyle": "<|-", "color": INK, "lw": 1.2})
    ax.text(0.02, 0.02, "order features per segment  ·  variable time gaps", color=SLATE, fontsize=6.6)


def mini_rul(ax):
    o = pd.read_csv(OOF)
    r = pd.read_csv(OOF_RAW)
    o = o[o["run_id"] == "Train1"].sort_values("mid_time_s")
    r = r[r["run_id"] == "Train1"].sort_values("mid_time_s")
    x = o["mid_time_s"].to_numpy() / 3600.0
    ax.fill_between(x, 0, o["predicted_rul_s"] / 3600.0, color=TEAL, alpha=0.10, lw=0)
    ax.plot(x, r["predicted_rul_s"].to_numpy() / 3600.0, color=CORAL, lw=0.9, alpha=0.85)
    ax.plot(x, o["predicted_rul_s"].to_numpy() / 3600.0, color=TEAL, lw=2.0, solid_capstyle="round")
    ax.plot(x, o["rul_s"].to_numpy() / 3600.0, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.set_ylim(bottom=0)
    _clean(ax)
    ax.text(0.03, 0.06, "RUL (h)", transform=ax.transAxes, color=SLATE, fontsize=6.6)


MINIS = [mini_vibration, mini_order, mini_transformer, mini_rul]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15.2, 5.4))
    canvas = fig.add_axes([0, 0, 1, 1])
    canvas.set_xlim(0, 100)
    canvas.set_ylim(0, 40)
    canvas.axis("off")

    canvas.text(1.5, 37.6, "Order-domain Transformer with causal end-of-life smoothing",
                fontsize=16, fontweight="bold", color=INK, va="center")
    canvas.text(1.5, 34.4, "Speed-robust bearing RUL: from raw vibration under varying RPM to a stable, causal RUL trajectory.",
                fontsize=10, color=SLATE, va="center")

    header_h = 5.2
    for i, ((num, title, sub, color), mini) in enumerate(zip(STAGES, MINIS)):
        x0 = CARD_X[i]
        canvas.add_patch(FancyBboxPatch((x0, CARD_Y), CARD_W, CARD_H,
                         boxstyle="round,pad=0.2,rounding_size=1.4",
                         facecolor=CARD_BG, edgecolor=color, linewidth=1.8, zorder=1))
        canvas.add_patch(FancyBboxPatch((x0, CARD_Y + CARD_H - header_h), CARD_W, header_h,
                         boxstyle="round,pad=0.2,rounding_size=1.4",
                         facecolor=color, edgecolor="none", zorder=2))
        canvas.text(x0 + 2.4, CARD_Y + CARD_H - header_h / 2, num, fontsize=12.5,
                    fontweight="bold", color="white", ha="center", va="center", zorder=3)
        canvas.text(x0 + 4.6, CARD_Y + CARD_H - header_h / 2, title, fontsize=10.5,
                    fontweight="bold", color="white", ha="left", va="center", zorder=3)
        canvas.text(x0 + CARD_W / 2, CARD_Y + 3.0, sub, fontsize=8.2, color=SLATE,
                    ha="center", va="center", zorder=3)
        iax = fig.add_axes(_ax_frac(canvas, x0 + 2.0, CARD_Y + 6.5, CARD_W - 4.0, CARD_H - header_h - 8.0))
        iax.set_zorder(3)
        mini(iax)

        if i < len(STAGES) - 1:
            xa = x0 + CARD_W
            canvas.add_patch(FancyArrowPatch((xa + 0.3, CARD_Y + CARD_H / 2),
                             (xa + 2.2, CARD_Y + CARD_H / 2),
                             arrowstyle="-|>", mutation_scale=16, lw=2.2, color=INK, zorder=4))

    canvas.text(50, 1.6,
                "Trained and evaluated leave-one-bearing-out without test-set leakage  ·  honest LOBO score 0.462",
                ha="center", va="center", fontsize=8.6, color=SLATE)

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"overview.{suffix}", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {OUT / 'overview.png'}")


if __name__ == "__main__":
    main()
