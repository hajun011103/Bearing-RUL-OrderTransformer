#!/usr/bin/env python
"""Animated hero for the README: time-frequency vs order domain under speed change.

As the shaft speed sweeps up, a fixed bearing-defect line moves in the
time-frequency (Hz) view and, integrated over the acquisition, smears into a
broad hump. In the order domain it stays a sharp line. The animation shows both
side by side, updating with the current RPM. Writes ``figures/results/rpm_to_order.gif``.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phmkorea_mplconfig")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

OUT = ROOT / "figures" / "results"
INK = "#1f2933"
MUTED = "#667085"
GRID = "#D0D5DD"
TEAL = "#2A9D8F"
NAVY = "#284B63"
RUST = "#C65D3A"

# (order, amplitude, label)
LINES = [(1.0, 1.0, "1x shaft"), (3.58, 0.72, "BPFO 3.58x"), (7.16, 0.36, "2x BPFO")]


def _spectrum(axis, centers, amps, sigma):
    out = np.zeros_like(axis)
    for c, a in zip(centers, amps):
        out += a * np.exp(-((axis - c) ** 2) / (2.0 * sigma**2))
    return out


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})

    freqs = np.linspace(0, 320, 700)
    orders = np.linspace(0, 10, 700)
    rpms = np.concatenate([np.linspace(1500, 2400, 52), np.full(10, 2400.0)])

    order_centers = [o for o, _, _ in LINES]
    amps = [a for _, a, _ in LINES]

    freq_cur, freq_acc, order_ref = [], [], _spectrum(orders, order_centers, amps, 0.045)
    acc_f = np.zeros_like(freqs)
    for rpm in rpms:
        f0 = rpm / 60.0
        fc = _spectrum(freqs, [o * f0 for o in order_centers], amps, 3.2)
        acc_f = np.maximum(acc_f, fc)
        freq_cur.append(fc.copy())
        freq_acc.append(acc_f.copy())

    fig, (ax_f, ax_o) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.17, wspace=0.22)

    def draw(i: int) -> None:
        rpm = rpms[i]
        f0 = rpm / 60.0
        for ax in (ax_f, ax_o):
            ax.clear()
            ax.grid(True, color=GRID, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylim(0, 1.25)
            ax.set_ylabel("Norm. amplitude")

        ax_f.fill_between(freqs, freq_acc[i], color=NAVY, alpha=0.14)
        ax_f.plot(freqs, freq_acc[i], color=NAVY, alpha=0.35, lw=1.0)
        ax_f.plot(freqs, freq_cur[i], color=NAVY, lw=1.8)
        ax_f.set_xlim(0, 320)
        ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_title("Time-frequency (fixed Hz)\nfault line moves & smears", fontsize=11, color=INK)
        for o in order_centers:
            ax_f.plot(o * f0, 0.02, marker="^", color=RUST, ms=7, clip_on=False)

        ax_o.fill_between(orders, order_ref, color=TEAL, alpha=0.16)
        ax_o.plot(orders, order_ref, color=TEAL, lw=1.9)
        ax_o.set_xlim(0, 10)
        ax_o.set_xlabel("Order (cycles / shaft rev)")
        ax_o.set_title("Order domain (shaft angle)\nfault line stays sharp", fontsize=11, color=INK)
        for o, _, name in LINES:
            ax_o.text(o, 1.06, name, ha="center", color=RUST, fontsize=8)

        fig.suptitle(f"Same bearing, shaft speed = {rpm:0.0f} rpm", x=0.5, y=0.97,
                     fontsize=13, fontweight="bold", color=INK)

    anim = FuncAnimation(fig, draw, frames=len(rpms), interval=90)
    OUT.mkdir(parents=True, exist_ok=True)
    anim.save(OUT / "rpm_to_order.gif", writer=PillowWriter(fps=11))
    plt.close(fig)
    print(f"wrote {OUT / 'rpm_to_order.gif'}")


if __name__ == "__main__":
    main()
