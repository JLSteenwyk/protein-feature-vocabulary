#!/usr/bin/env python3
"""SFig 10: Attribution Patching & Ablation Detail."""
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
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # Panel a: Attribution patching both models (absolute layers)
    ax = axes[0, 0]
    for model, path, color, n_layers in [
        ("ESM-3", RESULTS_SCALED / "attribution_patching.json", C_ESM3, 48),
        ("ESM-2", RESULTS_UNIFIED_ESM2 / "attribution_patching.json", C_ESM2, 33),
    ]:
        ap = load_json(path)
        layers = sorted(ap["per_layer"].keys(), key=int)
        lnums = [int(l) for l in layers]
        effects = [ap["per_layer"][l]["mean_effect"] for l in layers]
        ax.plot(lnums, effects, "-o", color=color, markersize=2, label=model, linewidth=1.2)
    ax.set_yscale("log"); ax.set_xlabel("Layer"); ax.set_ylabel("Mean attribution effect")
    ax.legend(fontsize=TICK_SIZE); ax.set_title("Attribution patching", fontsize=TITLE_SIZE)
    panel_label(ax, "a")

    # Panel b: Layer ablation both models
    ax = axes[0, 1]
    for model, path, color in [
        ("ESM-3", RESULTS_UNIFIED_ESM3 / "layer_ablation.json", C_ESM3),
        ("ESM-2", RESULTS_UNIFIED_ESM2 / "layer_ablation.json", C_ESM2),
    ]:
        la = load_json(path)
        layers = sorted(la["per_layer"].keys(), key=int)
        lnums = [int(l) for l in layers]
        kls = [la["per_layer"][l]["mean_kl"] for l in layers]
        ax.plot(lnums, kls, "-o", color=color, markersize=2, label=model, linewidth=1.2)
    ax.set_xlabel("Layer"); ax.set_ylabel("KL divergence (mean-ablated)")
    ax.legend(fontsize=TICK_SIZE); ax.set_title("Layer ablation", fontsize=TITLE_SIZE)
    panel_label(ax, "b")

    # Panel c: Head ablation ESM-3
    ax = axes[1, 0]
    try:
        ha3 = load_json(RESULTS_UNIFIED_ESM3 / "head_ablation.json")
        abl_layers = sorted(set(int(h["layer"]) for h in ha3["per_head"]))
        n_heads = ha3["n_heads"]
        mat = np.zeros((len(abl_layers), n_heads))
        layer_map = {l: i for i, l in enumerate(abl_layers)}
        for h in ha3["per_head"]:
            li = layer_map.get(int(h["layer"]))
            if li is not None and h["head"] < n_heads:
                mat[li, h["head"]] = h.get("mean_kl", 0)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(abl_layers)))
        ax.set_yticklabels(abl_layers, fontsize=6)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("Head ablation (ESM-3)", fontsize=TITLE_SIZE)
        plt.colorbar(im, ax=ax, shrink=0.7, label="KL")
    except Exception:
        ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes, ha="center")
    panel_label(ax, "c")

    # Panel d: Head ablation ESM-2
    ax = axes[1, 1]
    try:
        ha2 = load_json(RESULTS_UNIFIED_ESM2 / "head_ablation.json")
        abl_layers = sorted(set(int(h["layer"]) for h in ha2["per_head"]))
        n_heads = ha2["n_heads"]
        mat = np.zeros((len(abl_layers), n_heads))
        layer_map = {l: i for i, l in enumerate(abl_layers)}
        for h in ha2["per_head"]:
            li = layer_map.get(int(h["layer"]))
            if li is not None and h["head"] < n_heads:
                mat[li, h["head"]] = h.get("mean_kl", 0)
        im = ax.imshow(mat, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(abl_layers)))
        ax.set_yticklabels(abl_layers, fontsize=6)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer")
        ax.set_title("Head ablation (ESM-2)", fontsize=TITLE_SIZE)
        plt.colorbar(im, ax=ax, shrink=0.7, label="KL")
    except Exception:
        ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes, ha="center")
    panel_label(ax, "d")

    save_fig(fig, "sfig10_patching_ablation", supplementary=True)

if __name__ == "__main__":
    generate()
