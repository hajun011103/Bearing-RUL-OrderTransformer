#!/usr/bin/env python
"""Make the polished overview figures used in the README.

Produces two self-contained figures (no raw data needed):

* ``pipeline_overview.png`` — the four-stage method schematic.
* ``order_vs_time_demo.png`` — a synthetic swept-speed signal showing how a fixed
  fault frequency smears in the time-frequency spectrum but stays a sharp line in
  the order domain (uses the real ``order_spectrum`` from the package).
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from phm_pipeline.features import order_spectrum

OUT_DIR = ROOT / "figures" / "results"

INK = "#1f2933"
MUTED = "#667085"
GRID = "#D0D5DD"
TEAL = "#2A9D8F"
NAVY = "#284B63"
RUST = "#C65D3A"
GOLD = "#E9A93A"
STAGE_FILL = "#F4F7F9"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    stages = [
        (2.0, "1. Vibration + speed", "60 s acquisitions,\n25.6 kHz, varying RPM", NAVY),
        (26.0, "2. Order tracking", "resample to shaft angle;\nfault lines fixed in order", TEAL),
        (50.0, "3. Transformer", "time-gap-aware encoder\nover segment history", GOLD),
        (74.0, "4. Causal EOL smoothing", "running quantile of\npredicted end-of-life", RUST),
    ]
    w, h, y0 = 22.0, 22.0, 4.0
    inset_axes = []
    for x0, title, subtitle, color in stages:
        box = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=2.0, edgecolor=color, facecolor=STAGE_FILL,
        )
        ax.add_patch(box)
        ax.add_patch(FancyBboxPatch(
            (x0, y0 + h - 4.4), w, 4.4,
            boxstyle="round,pad=0.3,rounding_size=1.2",
            linewidth=0, facecolor=color,
        ))
        ax.text(x0 + w / 2, y0 + h - 2.2, title, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
        ax.text(x0 + w / 2, y0 + 2.6, subtitle, ha="center", va="center",
                color=MUTED, fontsize=8.5)
        iax = fig.add_axes(_ax_frac(ax, x0 + 2.4, y0 + 6.0, w - 4.8, h - 12.0))
        iax.set_xticks([])
        iax.set_yticks([])
        for s in iax.spines.values():
            s.set_visible(False)
        inset_axes.append((iax, color))

    for x0, _, _, _ in stages[:-1]:
        arrow = FancyArrowPatch(
            (x0 + w + 0.4, y0 + h / 2), (x0 + w + 3.6, y0 + h / 2),
            arrowstyle="-|>", mutation_scale=18, linewidth=2.2, color=INK,
        )
        ax.add_patch(arrow)

    # Mini-illustrations inside each stage.
    t = np.linspace(0, 1, 400)
    (a0, c0), (a1, c1), (a2, c2), (a3, c3) = inset_axes

    # 1. vibration + rpm
    a0.plot(t, np.sin(2 * np.pi * 9 * t) * (0.5 + 0.5 * t) + 0.08 * np.random.default_rng(0).standard_normal(t.size), color=c0, lw=0.9)
    a0.plot(t, 0.9 + 0.5 * t - 0.5, color=MUTED, lw=1.6)
    a0.set_ylim(-1.4, 1.4)

    # 2. order spectrum sharp lines
    orders = np.linspace(0, 10, 500)
    spec = np.zeros_like(orders)
    for o, amp in [(1.0, 1.0), (3.58, 0.75), (7.16, 0.4)]:
        spec += amp * np.exp(-((orders - o) ** 2) / 0.004)
    a1.plot(orders, spec, color=c1, lw=1.4)
    a1.fill_between(orders, spec, color=c1, alpha=0.18)
    a1.set_ylim(0, 1.2)

    # 3. transformer: stacked tokens + attention
    rng = np.random.default_rng(1)
    for row in range(4):
        a2.plot([0, 1], [row, row], color=c2, lw=3, alpha=0.85)
    for _ in range(7):
        i, j = rng.integers(0, 4, 2)
        a2.plot([0.15, 0.85], [i, j], color=INK, lw=0.5, alpha=0.35)
    a2.set_ylim(-0.6, 3.6)
    a2.set_xlim(-0.1, 1.1)

    # 4. jittery RUL -> smoothed decline
    x = np.linspace(0, 1, 120)
    true = 1 - x
    raw = true + 0.12 * np.sin(25 * x) + 0.05 * np.random.default_rng(2).standard_normal(x.size)
    smooth = true + 0.02 * np.cos(6 * x)
    a3.plot(x, raw, color=MUTED, lw=0.8, alpha=0.7)
    a3.plot(x, smooth, color=c3, lw=2.0)
    a3.plot(x, true, color=INK, lw=1.2, ls="--")
    a3.set_ylim(-0.15, 1.15)

    fig.suptitle("Order-domain Transformer with causal end-of-life smoothing",
                 x=0.012, y=1.02, ha="left", fontsize=15, fontweight="bold", color=INK)
    _save(fig, "pipeline_overview")


def _ax_frac(parent_ax, x, y, w, h):
    """Convert data-space rect in ``parent_ax`` to a figure-fraction rect."""
    fig = parent_ax.figure
    p0 = parent_ax.transData.transform((x, y))
    p1 = parent_ax.transData.transform((x + w, y + h))
    inv = fig.transFigure.inverted()
    f0 = inv.transform(p0)
    f1 = inv.transform(p1)
    return [f0[0], f0[1], f1[0] - f0[0], f1[1] - f0[1]]


def order_vs_time_demo() -> None:
    fs = 25_600.0
    duration = 2.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(7)

    # Shaft speed sweeps during the acquisition (this is what smears the FFT).
    rpm = np.linspace(1500.0, 2400.0, n)
    f0 = rpm / 60.0
    phase = 2.0 * np.pi * np.cumsum(f0) / fs  # instantaneous shaft phase

    fault_order = 3.58  # BPFO in shaft orders
    signal = (
        1.0 * np.sin(phase)                                   # 1x shaft
        + 0.7 * np.sin(fault_order * phase)                   # fault line
        + 0.35 * np.sin(2 * fault_order * phase)              # fault harmonic
        + 0.15 * rng.standard_normal(n)
    )

    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft(signal * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = spectrum / spectrum.max()

    orders, amp = order_spectrum(
        signal, rpm, sample_rate_hz=fs, samples_per_revolution=128, max_order=12.0
    )
    amp = amp / (amp.max() + 1e-12)

    _style()
    fig, (ax_t, ax_o) = plt.subplots(2, 1, figsize=(9.6, 6.4))

    keep = freqs <= 260.0
    ax_t.plot(freqs[keep], spectrum[keep], color=NAVY, lw=1.3)
    ax_t.fill_between(freqs[keep], spectrum[keep], color=NAVY, alpha=0.15)
    ax_t.set_title("Time-frequency spectrum (fixed-Hz FFT) — fault lines smear under speed change",
                   fontsize=12, color=INK)
    ax_t.set_xlabel("Frequency (Hz)")
    ax_t.set_ylabel("Norm. amplitude")
    ax_t.grid(True, color=GRID, lw=0.5)
    ax_t.spines[["top", "right"]].set_visible(False)
    ax_t.text(0.98, 0.9, "shaft speed 1500 → 2400 rpm", transform=ax_t.transAxes,
              ha="right", color=MUTED, fontsize=9)

    ax_o.plot(orders, amp, color=TEAL, lw=1.5)
    ax_o.fill_between(orders, amp, color=TEAL, alpha=0.18)
    for o, name in [(1.0, "1× shaft"), (fault_order, "BPFO 3.58×"), (2 * fault_order, "2× BPFO")]:
        ax_o.axvline(o, color=RUST, lw=1.0, ls=":")
        ax_o.text(o, 1.04, name, ha="center", color=RUST, fontsize=8.5)
    ax_o.set_ylim(0, 1.18)
    ax_o.set_title("Order-domain spectrum (angular resampling) — fault lines stay sharp",
                   fontsize=12, color=INK)
    ax_o.set_xlabel("Order (cycles per shaft revolution)")
    ax_o.set_ylabel("Norm. amplitude")
    ax_o.grid(True, color=GRID, lw=0.5)
    ax_o.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save(fig, "order_vs_time_demo")


def main() -> None:
    _style()
    pipeline_overview()
    order_vs_time_demo()
    print(f"wrote overview figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
