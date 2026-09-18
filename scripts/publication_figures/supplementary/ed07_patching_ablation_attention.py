#!/usr/bin/env python3
"""ED7: Attribution Patching, Layer Ablation & Attention Atlas."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from config import *; from style import setup_style; from utils import load_json, save_fig, panel_label
setup_style()

def generate():
    fig = plt.figure(figsize=(7.2, 6.5))
    gs = GridSpec(2, 4, height_ratios=[1, 1.3], hspace=0.45, wspace=0.40)

    # Panel a: Attribution patching (both models)
    ax = fig.add_subplot(gs[0, :2])
    for model, path, color in [("ESM-3", RESULTS_SCALED / "attribution_patching.json", C_ESM3),
                                ("ESM-2", RESULTS_UNIFIED_ESM2 / "attribution_patching.json", C_ESM2)]:
        ap = load_json(path); layers = sorted(ap["per_layer"].keys(), key=int)
        ax.semilogy([int(l) for l in layers], [ap["per_layer"][l]["mean_effect"] for l in layers],
                    "-o", color=color, markersize=2, label=model, lw=1.2)
    ax.set_xlabel("Layer"); ax.set_ylabel("Attribution effect")
    ax.legend(fontsize=TICK_SIZE, frameon=False)
    panel_label(ax, "a")

    # Panel b: Layer ablation (both models)
    ax = fig.add_subplot(gs[0, 2:])
    for model, path, color in [("ESM-3", RESULTS_UNIFIED_ESM3 / "layer_ablation.json", C_ESM3),
                                ("ESM-2", RESULTS_UNIFIED_ESM2 / "layer_ablation.json", C_ESM2)]:
        la = load_json(path); layers = sorted(la["per_layer"].keys(), key=int)
        ax.plot([int(l) for l in layers], [la["per_layer"][l]["mean_kl"] for l in layers],
                "-o", color=color, markersize=2, label=model, lw=1.2)
    ax.set_xlabel("Layer"); ax.set_ylabel("KL (ablated)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=TICK_SIZE, frameon=False)
    panel_label(ax, "b")

    # Row 2: full attention atlas (spans most of width)
    cmap = LinearSegmentedColormap.from_list("v", VIRIDIS, N=256)
    ax_main = fig.add_subplot(gs[1, :3])
    atlas = load_json(RESULTS_SCALED / "attention_atlas.json")
    jsd = np.array(atlas["mean_jsd_matrix"])
    im = ax_main.imshow(jsd.T, aspect="auto", cmap=cmap, origin="lower", interpolation="nearest")
    ax_main.set_xlabel("Layer"); ax_main.set_ylabel("Head"); ax_main.set_xticks(range(0, 48, 4))
    panel_label(ax_main, "c", x=-0.08)

    ax_cb = fig.add_subplot(gs[1, 3])
    ax_cb.axis("off")
    fig.colorbar(im, ax=ax_cb, fraction=0.8, location="left", label="JSD")

    save_fig(fig, "ed07_patching_ablation_attention", supplementary=True)

if __name__ == "__main__": generate()
