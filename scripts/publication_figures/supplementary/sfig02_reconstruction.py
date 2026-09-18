#!/usr/bin/env python3
"""SFig 2: Reconstruction Quality Detail."""
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
    rq = load_json(RESULTS_SCALED / "reconstruction_quality.json")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.5))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    labels = []
    cos_vals, cos_stds, ve_vals, ve_stds, mse_vals, mse_stds = [], [], [], [], [], []
    for model in ["esm3", "esm2"]:
        for level in ["residue", "protein"]:
            if level in rq[model]:
                r = rq[model][level]
                labels.append(f"{model.upper()}\n{level}")
                cos_vals.append(r["cosine_similarity"]["mean"])
                cos_stds.append(r["cosine_similarity"]["std"])
                ve_vals.append(r["variance_explained"]["mean"])
                ve_stds.append(r["variance_explained"]["std"])
                mse_vals.append(r["mse"]["mean"])
                mse_stds.append(r["mse"]["std"])

    colors = [C_ESM3, C_ESM3, C_ESM2, C_ESM2]
    x = np.arange(len(labels))

    # Panel a: Cosine
    ax = axes[0, 0]
    ax.bar(x, cos_vals, yerr=cos_stds, color=colors, capsize=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Cosine similarity"); ax.set_ylim(0.85, 1.0)
    ax.set_title("Cosine similarity", fontsize=TITLE_SIZE)
    panel_label(ax, "a")

    # Panel b: Variance explained
    ax = axes[0, 1]
    ax.bar(x, ve_vals, yerr=ve_stds, color=colors, capsize=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("Variance explained"); ax.set_ylim(0.75, 1.0)
    ax.set_title("Variance explained", fontsize=TITLE_SIZE)
    panel_label(ax, "b")

    # Panel c: MSE
    ax = axes[1, 0]
    ax.bar(x, mse_vals, yerr=mse_stds, color=colors, capsize=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("MSE"); ax.set_yscale("log")
    ax.set_title("Mean squared error", fontsize=TITLE_SIZE)
    panel_label(ax, "c")

    # Panel d: Percentile comparison (cosine, residue level)
    ax = axes[1, 1]
    pctiles = ["5", "25", "50", "75", "95"]
    for model, color, marker in [("esm3", C_ESM3, "o"), ("esm2", C_ESM2, "s")]:
        if "residue" in rq[model]:
            p = rq[model]["residue"]["cosine_similarity"]["percentiles"]
            vals = [p[k] for k in pctiles]
            ax.plot(pctiles, vals, f"-{marker}", color=color, label=model.upper(), markersize=4)
    ax.set_xlabel("Percentile"); ax.set_ylabel("Cosine similarity")
    ax.legend(fontsize=TICK_SIZE); ax.set_title("Cosine percentiles (residue)", fontsize=TITLE_SIZE)
    panel_label(ax, "d")

    save_fig(fig, "sfig02_reconstruction", supplementary=True)

if __name__ == "__main__":
    generate()
