#!/usr/bin/env python3
"""ED3: Negative Controls & Robustness (random baseline + homology splits + same-protein attribution)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import *; from style import setup_style; from utils import load_json, save_fig, panel_label
setup_style()

def generate():
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.60, wspace=0.45,
                           left=0.08, right=0.97, top=0.96, bottom=0.08)

    rb = load_json(RESULTS_SCALED / "random_baseline.json")

    # Panel a: AUROC strip plot
    ax = fig.add_subplot(gs[0, 0])
    for col, (model, mc) in enumerate([("esm3", C_ESM3), ("esm2", C_ESM2)]):
        d = rb[model]
        x_base = col * 1.5
        sae_auroc = d["sae"]["probing"]["auroc"]
        rand_aurocs = [r["probing"]["auroc"] for r in d["random_projection"]]
        shuf_aurocs = [r["probing"]["auroc"] for r in d["shuffled"]]
        ax.axhline(sae_auroc, xmin=col*0.5, xmax=(col+1)*0.5, color=C_SAE, lw=1.5)
        ax.scatter([x_base+0.3]*len(rand_aurocs), rand_aurocs, color=C_RANDOM, s=20, zorder=5)
        ax.scatter([x_base+0.7]*len(shuf_aurocs), shuf_aurocs, color=C_SHUFFLED, s=20, zorder=5)
    ax.axhline(0.5, color="grey", ls="--", lw=0.5)
    ax.set_ylabel("AUROC")
    ax.set_xticks([0.5, 2.0]); ax.set_xticklabels(["ESM-3", "ESM-2"])
    panel_label(ax, "a")

    # Panel b: GO terms comparison
    ax = fig.add_subplot(gs[0, 1])
    for i, (model, mc) in enumerate([("esm3", C_ESM3), ("esm2", C_ESM2)]):
        s = rb[model]["summary"]
        x = np.array([0, 1, 2]) + i * 3.5
        vals = [s["sae_go_mean_terms"], s["random_go_mean_terms"], s["shuffled_go_mean_terms"]]
        colors = [C_SAE, C_RANDOM, C_SHUFFLED]
        ax.bar(x, vals, color=colors, edgecolor="white")
        for j, v in enumerate(vals):
            ax.text(x[j], v + 1, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_ylabel("GO terms/feature")
    ax.set_xticks([1, 4.5]); ax.set_xticklabels(["ESM-3", "ESM-2"])
    panel_label(ax, "b")

    # Panel c: Same-protein attribution comparison
    ax = fig.add_subplot(gs[0, 2])
    try:
        sp = load_json(RESULTS_SCALED / "same_protein_causal.json")
        cp = load_json(RESULTS_SCALED / "attribution_patching.json")
        cp_layers = sorted(int(k) for k in cp["per_layer"])
        cp_effects = [cp["per_layer"][str(l)]["mean_effect"] for l in cp_layers]
        cp_depth = [l / (cp["n_layers"] - 1) for l in cp_layers]
        sp_attr = sp["same_protein_attribution"]
        sp_layers = sorted(int(k) for k in sp_attr["per_layer"])
        sp_effects = [sp_attr["per_layer"][str(l)]["mean_effect"] for l in sp_layers]
        sp_depth = [l / 47 for l in sp_layers]
        ax.semilogy(cp_depth, cp_effects, "-o", color=C_ESM3, markersize=2, lw=1.2, label="Cross-protein")
        ax.semilogy(sp_depth, sp_effects, "--^", color=C_ESM2, markersize=2, lw=1, alpha=0.7, label="Same-protein")
        ax.set_xlabel("Relative depth"); ax.set_ylabel("Attribution effect")
        ax.legend(fontsize=6, frameon=False)
    except Exception:
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
    panel_label(ax, "c")

    # Row 2: Homology splits
    try:
        hs = load_json(RESULTS_SCALED / "homology_splits.json")

        # Panel d: Probing AUROC per fold
        ax = fig.add_subplot(gs[1, 0])
        folds = hs["probing"]["per_fold"]
        aurocs = [f["auroc"] for f in folds]
        ax.bar(range(len(aurocs)), aurocs, color=C_ESM3, edgecolor="white")
        ax.axhline(hs["probing"]["original_auroc"], color="red", ls="--", lw=1, label=f"Original: {hs['probing']['original_auroc']:.3f}")
        mean_a = hs["probing"]["mean_auroc"]
        ax.axhline(mean_a, color="black", ls=":", lw=1, label=f"Mean: {mean_a:.3f}")
        for i, a in enumerate(aurocs):
            ax.text(i, a / 2, f"{a:.3f}", ha="center", va="center", fontsize=6, color="white")
        ax.set_xlabel("Fold"); ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, frameon=False, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.35))
        panel_label(ax, "d")

        # Panel e: Convergence per fold
        ax = fig.add_subplot(gs[1, 1])
        cfolds = hs["convergence"]["per_fold"]
        fracs = [f["frac_above_0.3"] for f in cfolds]
        ax.bar(range(len(fracs)), fracs, color=C_ESM3, edgecolor="white")
        ax.axhline(hs["convergence"]["original_frac"], color="red", ls="--", lw=1, label=f"Full set: {hs['convergence']['original_frac']:.2f}")
        mean_f = hs["convergence"]["mean_frac_above_0.3"]
        ax.axhline(mean_f, color="black", ls=":", lw=1, label=f"Mean: {mean_f:.2f}")
        for i, f in enumerate(fracs):
            ax.text(i, f / 2, f"{f:.2f}", ha="center", va="center", fontsize=6, color="white")
        ax.set_xlabel("Fold"); ax.set_ylabel("Frac > 0.3"); ax.set_ylim(0, 1.0)
        ax.legend(fontsize=6, frameon=False, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.35))
        panel_label(ax, "e")

    except Exception:
        fig.add_subplot(gs[1, 0]).text(0.5, 0.5, "Homology splits\nnot available", transform=plt.gca().transAxes, ha="center")
        fig.add_subplot(gs[1, 1]).text(0.5, 0.5, "N/A", transform=plt.gca().transAxes, ha="center")

    # Hide unused bottom-right panel
    fig.add_subplot(gs[1, 2]).axis("off")

    save_fig(fig, "ed03_negative_controls", supplementary=True)

if __name__ == "__main__":
    generate()
