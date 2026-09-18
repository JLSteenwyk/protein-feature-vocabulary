#!/usr/bin/env python3
"""SFig 8: Cross-Modal Feature Detail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label

setup_style()

def generate():
    cm = load_json(RESULTS_SCALED / "cross_modal_features.json")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.8))
    fig.subplots_adjust(wspace=0.40)

    # Collect Cohen's d values
    enh = cm.get("top_enhanced", [])
    sup = cm.get("top_suppressed", [])
    enh_d = [f["cohens_d"] for f in enh if "cohens_d" in f]
    sup_d = [f["cohens_d"] for f in sup if "cohens_d" in f]

    # Panel a: Cohen's d distribution
    ax = axes[0]
    if enh_d:
        ax.hist(enh_d, bins=25, color=C_ENHANCED, alpha=0.7, label="Enhanced", edgecolor="white")
    if sup_d:
        ax.hist(sup_d, bins=25, color=C_SUPPRESSED, alpha=0.7, label="Suppressed", edgecolor="white")
    ax.axvline(0, color="black", ls="-", lw=0.5)
    ax.set_xlabel("Cohen's d"); ax.set_ylabel("Count")
    ax.legend(fontsize=6, frameon=False)
    panel_label(ax, "a")

    # Panel b: Cohen's d vs mean activation
    ax = axes[1]
    for feat_list, color, label in [(enh, C_ENHANCED, "Enhanced"), (sup, C_SUPPRESSED, "Suppressed")]:
        d_vals = [f.get("cohens_d", 0) for f in feat_list]
        mean_deltas = [f.get("mean_delta", 0) for f in feat_list]
        ax.scatter(d_vals, mean_deltas, color=color, alpha=0.5, s=10, label=label, rasterized=True)
    ax.axhline(0, color="grey", ls="--", lw=0.5)
    ax.axvline(0, color="grey", ls="--", lw=0.5)
    ax.set_xlabel("Cohen's d"); ax.set_ylabel("Mean activation change\n(S+St \u2212 S)")
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    panel_label(ax, "b")

    # Panel c: Effect size summary
    ax = axes[2]
    categories = ["Enhanced", "Suppressed"]
    medians = [np.median(enh_d) if enh_d else 0, np.median(sup_d) if sup_d else 0]
    colors_c = [C_ENHANCED, C_SUPPRESSED]
    ax.bar(range(2), medians, color=colors_c, edgecolor="white")
    ax.set_xticks(range(2)); ax.set_xticklabels(categories)
    ax.set_ylabel("Median Cohen's d")
    ax.set_ylim(min(medians) - 0.5, max(medians) + 0.5)
    ax.axhline(0, color="black", lw=0.5)
    for i, v in enumerate(medians):
        offset = 0.1 if v > 0 else -0.2
        ax.text(i, v + offset, f"{v:.3f}", ha="center", fontsize=6)
    panel_label(ax, "c")

    save_fig(fig, "sfig08_cross_modal", supplementary=True)

if __name__ == "__main__":
    generate()
