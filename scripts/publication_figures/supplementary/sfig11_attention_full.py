#!/usr/bin/env python3
"""SFig 11: Attention Atlas Full Resolution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label

setup_style()

def generate():
    atlas = load_json(RESULTS_SCALED / "attention_atlas.json")
    jsd_matrix = np.array(atlas["mean_jsd_matrix"])  # (48, 24)

    fig = plt.figure(figsize=(DOUBLE_COL, 8.0))
    gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5],
                  hspace=0.05, wspace=0.05)

    # Main heatmap
    ax_main = fig.add_subplot(gs[1, 0])
    cmap = LinearSegmentedColormap.from_list("viridis_custom", VIRIDIS, N=256)
    im = ax_main.imshow(jsd_matrix.T, aspect="auto", cmap=cmap,
                        origin="lower", interpolation="nearest")
    ax_main.set_xlabel("Layer")
    ax_main.set_ylabel("Head")
    ax_main.set_xticks(range(0, 48, 4))
    ax_main.set_yticks(range(0, 24, 4))

    # Top marginal: mean JSD per head
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    mean_per_head = jsd_matrix.mean(axis=0)  # (24,)
    # Actually we want mean per layer for top marginal
    mean_per_layer = jsd_matrix.mean(axis=1)  # (48,)
    ax_top.bar(range(48), mean_per_layer, color=C_ESM3, width=0.8)
    ax_top.set_ylabel("Mean JSD")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_title("ESM-3 Attention Structure-Responsiveness (JSD)", fontsize=TITLE_SIZE)

    # Right marginal: mean JSD per head
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    mean_per_head = jsd_matrix.mean(axis=0)  # (24,)
    ax_right.barh(range(24), mean_per_head, color=C_ESM3, height=0.8)
    ax_right.set_xlabel("Mean JSD")
    ax_right.tick_params(labelleft=False)

    # Colorbar
    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar_ax.axis("off")
    cbar = fig.colorbar(im, ax=cbar_ax, fraction=0.8, pad=0.05, location="left")
    cbar.set_label("JSD", fontsize=AXIS_LABEL_SIZE)

    # Responsive head info
    n_resp = atlas["n_responsive_heads"]
    peak = atlas.get("peak_layer", "?")
    fig.text(0.02, 0.98, f"a", fontsize=PANEL_LABEL_SIZE, fontweight="bold",
             va="top", ha="left")
    fig.text(0.15, 0.98, f"{n_resp} responsive heads (JSD > {atlas['jsd_threshold']}), peak at layer {peak}",
             fontsize=TICK_SIZE, va="top")

    save_fig(fig, "sfig11_attention_full", supplementary=True)

if __name__ == "__main__":
    generate()
