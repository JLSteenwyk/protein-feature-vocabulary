#!/usr/bin/env python3
"""Graphical abstract: three key findings."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from config import *
from style import setup_style
from utils import load_json, save_fig

setup_style()


def generate():
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45,
                           left=0.06, right=0.97, top=0.85, bottom=0.12)

    blue = C_ESM3
    red = C_ESM2
    enhanced_c = C_ENHANCED
    suppressed_c = C_SUPPRESSED
    invariant_c = C_INVARIANT

    # === Panel 1: Venn diagram — 78% convergence ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-1.5, 1.5)
    ax1.axis("off")
    ax1.set_title("Feature convergence", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

    circle1 = Circle((-0.35, 0), 1.0, facecolor=blue, alpha=0.2,
                      edgecolor=blue, linewidth=1.2)
    circle2 = Circle((0.35, 0), 1.0, facecolor=red, alpha=0.2,
                      edgecolor=red, linewidth=1.2)
    ax1.add_patch(circle1)
    ax1.add_patch(circle2)
    ax1.text(0, 0, "78%", ha="center", va="center",
             fontsize=16, fontweight="bold", color="#333333")
    ax1.text(-1.0, 0, "22%", ha="center", va="center",
             fontsize=9, color=blue, alpha=0.7)
    ax1.text(1.0, 0, "33%", ha="center", va="center",
             fontsize=9, color=red, alpha=0.7)
    ax1.text(-0.85, 1.1, "ESM-3", ha="center", va="center",
             fontsize=7, color=blue, fontweight="bold")
    ax1.text(0.85, 1.1, "ESM-2", ha="center", va="center",
             fontsize=7, color=red, fontweight="bold")
    ax1.text(0, -1.35, "Convergent features\ncarry the biology\n(AUROC 0.925 vs 0.661)",
             ha="center", va="center", fontsize=6, color="#555555")

    # === Panel 2: The paradox — enhanced features MORE convergent ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("The structure paradox", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

    # Violin-like simplified bars
    categories = ["Enhanced", "Suppressed", "Invariant"]
    medians = [0.515, 0.494, 0.442]
    colors = [enhanced_c, suppressed_c, invariant_c]
    bars = ax2.bar(range(3), medians, color=colors, edgecolor="white", width=0.6)

    for i, (bar, med) in enumerate(zip(bars, medians)):
        ax2.text(i, med + 0.015, f"{med:.3f}", ha="center", va="bottom",
                 fontsize=6, color="#333333")

    ax2.set_xticks(range(3))
    ax2.set_xticklabels(["Enh.", "Supp.", "Inv."], fontsize=TICK_SIZE)
    ax2.set_ylabel("Convergence\n(best-match r)", fontsize=TICK_SIZE)
    ax2.set_ylim(0, 0.6)
    ax2.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Annotation arrow highlighting the paradox
    ax2.annotate("Structure-responsive\nfeatures converge\nMORE with seq-only",
                 xy=(0, 0.515), xytext=(1.5, 0.56),
                 fontsize=5.5, ha="center", color=enhanced_c,
                 arrowprops=dict(arrowstyle="->", color=enhanced_c, lw=0.8))

    # === Panel 3: L0H7 bottleneck ===
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("L0H7: structure bottleneck", fontsize=AXIS_LABEL_SIZE, fontweight="bold")

    # Grouped bars: L0H7 vs random heads
    measures = ["SS3", "Token", "Enh.\nSAE"]
    l0h7_vals = [0.60, 0.69, 0.72]  # fraction remaining
    random_vals = [0.83, 0.99, 0.999]

    x = np.arange(len(measures))
    width = 0.3

    from pypubfigs.palettes import friendly_pal
    ito = friendly_pal("ito_seven")

    ax3.bar(x - width / 2, l0h7_vals, width, color=ito[1],
            edgecolor="white", linewidth=0.3, label="L0H7")
    ax3.bar(x + width / 2, random_vals, width, color=ito[4],
            edgecolor="white", linewidth=0.3, label="Random L0")

    # Loss annotations on L0H7 bars
    for i, val in enumerate(l0h7_vals):
        pct_lost = (1 - val) * 100
        ax3.text(x[i] - width / 2, val + 0.02,
                 f"-{pct_lost:.0f}%", ha="center", va="bottom",
                 fontsize=6, color="#CC3311")

    ax3.set_xticks(x)
    ax3.set_xticklabels(measures, fontsize=TICK_SIZE)
    ax3.set_ylabel("Fraction remaining", fontsize=TICK_SIZE)
    ax3.set_ylim(0, 1.15)
    ax3.axhline(1.0, color="#CCCCCC", ls="--", lw=0.5)
    ax3.legend(fontsize=6, loc="lower center", frameon=False, ncol=2,
               bbox_to_anchor=(0.5, -0.22))

    save_fig(fig, "graphical_abstract")


if __name__ == "__main__":
    generate()
