#!/usr/bin/env python3
"""SFig 1: SAE Training & Architecture."""
import sys
from pathlib import Path
_src_dir = str(Path(__file__).parent.parent.parent.parent / "src")
_fig_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, _src_dir)
sys.path.insert(0, _fig_dir)  # must be first so our utils beats src/utils

import numpy as np
import torch
import matplotlib.pyplot as plt
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label

setup_style()

def generate():
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    # Panel a: Training loss curves
    ax = axes[0, 0]
    for model, tag, color in [("esm3", "esm3_residue_ef8_k64", C_ESM3),
                               ("esm2", "esm2_residue_ef8_k64", C_ESM2)]:
        ckpt_path = MODEL_ROOT / tag / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            lh = ckpt.get("log_history", [])
            steps = [e["step"] for e in lh]
            losses = [e["loss"] for e in lh]
            ax.plot(steps, losses, color=color, label=model.upper(), linewidth=1.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=TICK_SIZE)
    ax.set_title("Training loss", fontsize=TITLE_SIZE)
    panel_label(ax, "a")

    # Panel b: Cosine similarity per architecture
    ax = axes[0, 1]
    x_pos = 0
    xticks, xlabels = [], []
    for model, color in [("esm3", C_ESM3), ("esm2", C_ESM2)]:
        ts = load_json(MODEL_ROOT / f"{model}_sae_training_summary.json")
        for entry in ts:
            short = entry["tag"].replace(f"{model}_", "")
            ax.bar(x_pos, entry["cosine"], color=color, width=0.7, edgecolor="white")
            xticks.append(x_pos)
            xlabels.append(short)
            x_pos += 1
        x_pos += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Reconstruction quality", fontsize=TITLE_SIZE)
    panel_label(ax, "b")

    # Panel c: Dead feature fraction
    ax = axes[1, 0]
    x_pos = 0
    xticks, xlabels = [], []
    for model, color in [("esm3", C_ESM3), ("esm2", C_ESM2)]:
        ts = load_json(MODEL_ROOT / f"{model}_sae_training_summary.json")
        for entry in ts:
            short = entry["tag"].replace(f"{model}_", "")
            ax.bar(x_pos, entry["dead_frac"] * 100, color=color, width=0.7, edgecolor="white")
            xticks.append(x_pos)
            xlabels.append(short)
            x_pos += 1
        x_pos += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Dead features (%)")
    ax.set_title("Dead feature fraction", fontsize=TITLE_SIZE)
    panel_label(ax, "c")

    # Panel d: L0 distribution
    ax = axes[1, 1]
    sp = load_json(RESULTS_SCALED / "feature_sparsity_analysis.json")
    for model, color in [("esm3", C_ESM3), ("esm2", C_ESM2)]:
        h = sp[model]["residue"]["l0_distribution"]["histogram"]
        counts = np.array(h["counts"])
        edges = np.array(h["bin_edges"])
        centers = (edges[:-1] + edges[1:]) / 2
        ax.step(centers, counts, color=color, label=model.upper(), linewidth=1.2, where="mid")
    ax.set_xlabel("Features per token (L0)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=TICK_SIZE)
    ax.set_title("L0 distribution", fontsize=TITLE_SIZE)
    panel_label(ax, "d")

    fig.suptitle("Supplementary Figure 1: SAE Training & Architecture", fontsize=TITLE_SIZE, y=1.02)
    save_fig(fig, "sfig01_training", supplementary=True)

if __name__ == "__main__":
    generate()
