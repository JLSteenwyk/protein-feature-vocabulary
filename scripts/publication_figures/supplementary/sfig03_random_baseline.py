#!/usr/bin/env python3
"""SFig 3: Random Baseline Detail."""
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
    rb = load_json(RESULTS_SCALED / "random_baseline.json")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for col, model in enumerate(["esm3", "esm2"]):
        d = rb[model]
        sae_auroc = d["sae"]["probing"]["auroc"]
        rand_aurocs = [r["probing"]["auroc"] for r in d["random_projection"]]
        shuf_aurocs = [r["probing"]["auroc"] for r in d["shuffled"]]

        sae_go = d["sae"]["go_enrichment"]["mean_enriched_terms_per_feature"]
        rand_go = [r["go_enrichment"]["mean_enriched_terms_per_feature"] for r in d["random_projection"]]
        shuf_go = [r["go_enrichment"]["mean_enriched_terms_per_feature"] for r in d["shuffled"]]

        sae_or = d["sae"]["go_enrichment"]["median_best_odds_ratio"]
        rand_or = [r["go_enrichment"]["median_best_odds_ratio"] for r in d["random_projection"]]
        shuf_or = [r["go_enrichment"]["median_best_odds_ratio"] for r in d["shuffled"]]

        sae_f3 = d["sae"]["go_enrichment"]["frac_with_3plus_terms"]
        rand_f3 = [r["go_enrichment"]["frac_with_3plus_terms"] for r in d["random_projection"]]
        shuf_f3 = [r["go_enrichment"]["frac_with_3plus_terms"] for r in d["shuffled"]]

        mc = C_ESM3 if model == "esm3" else C_ESM2

        # Panel a/b: AUROC strip plot
        ax = axes[0, col]
        ax.axhline(sae_auroc, color=C_SAE, linewidth=1.5, label="SAE")
        ax.scatter(np.zeros(len(rand_aurocs)) + 0.3, rand_aurocs, color=C_RANDOM, s=30, zorder=5, label="Random")
        ax.scatter(np.zeros(len(shuf_aurocs)) + 0.7, shuf_aurocs, color=C_SHUFFLED, s=30, zorder=5, label="Shuffled")
        ax.axhline(0.5, color="grey", ls="--", lw=0.5)
        ax.set_xlim(-0.2, 1.2); ax.set_xticks([0.3, 0.7]); ax.set_xticklabels(["Random", "Shuffled"])
        ax.set_ylabel("AUROC"); ax.set_title(f"{model.upper()} — Probing AUROC", fontsize=TITLE_SIZE)
        ax.legend(fontsize=6, loc="center right")
        if col == 0: panel_label(ax, "a")
        else: panel_label(ax, "b")

        # Panel c/d: GO terms + odds ratio
        ax = axes[1, col]
        x = np.arange(3)
        vals = [sae_go, np.mean(rand_go), np.mean(shuf_go)]
        errs = [0, np.std(rand_go), np.std(shuf_go)]
        colors_bar = [C_SAE, C_RANDOM, C_SHUFFLED]
        ax.bar(x, vals, yerr=errs, color=colors_bar, capsize=3, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(["SAE", "Random", "Shuffled"])
        ax.set_ylabel("GO terms / feature (mean)"); ax.set_title(f"{model.upper()} — GO enrichment", fontsize=TITLE_SIZE)
        for i, v in enumerate(vals):
            ax.text(i, v + errs[i] + 0.5, f"{v:.1f}", ha="center", fontsize=6)
        if col == 0: panel_label(ax, "c")
        else: panel_label(ax, "d")

    save_fig(fig, "sfig03_random_baseline", supplementary=True)

if __name__ == "__main__":
    generate()
