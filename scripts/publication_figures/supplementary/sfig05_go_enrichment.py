#!/usr/bin/env python3
"""SFig 5: GO Enrichment Detail."""
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
    go = load_json(RESULTS_SCALED / "go_enrichment.json")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    for col, (model, color) in enumerate([("esm3", C_ESM3), ("esm2", C_ESM2)]):
        d = go[model]
        pfr = d.get("per_feature_results", [])

        # Count enriched terms per feature
        n_terms = [f.get("n_enriched_terms", len(f.get("enriched_terms", []))) for f in pfr]
        n_terms = np.array(n_terms)

        # Panel a/b: Distribution of enriched terms
        ax = axes[0, col]
        ax.hist(n_terms[n_terms > 0], bins=30, color=color, alpha=0.8, edgecolor="white")
        med = np.median(n_terms[n_terms > 0]) if (n_terms > 0).any() else 0
        ax.axvline(med, color="black", ls="--", lw=1)
        ax.set_xlabel("Enriched GO terms per feature")
        ax.set_ylabel("Count")
        ax.set_title(f"{model.upper()} — GO terms/feature", fontsize=TITLE_SIZE)
        frac = d.get("frac_enriched", (n_terms > 0).mean())
        ax.text(0.95, 0.95, f"{frac*100:.1f}% enriched\nmedian={med:.0f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        panel_label(ax, "a" if col == 0 else "b")

    # Panel c: Top features by enrichment count (ESM-3)
    ax = axes[1, 0]
    pfr3 = go["esm3"].get("per_feature_results", [])
    top = sorted(pfr3, key=lambda f: -f.get("n_enriched_terms", len(f.get("enriched_terms", []))))[:20]
    names = [f"F{f.get('feature_id', i)}" for i, f in enumerate(top)]
    counts = [f.get("n_enriched_terms", len(f.get("enriched_terms", []))) for f in top]
    y = np.arange(len(names))
    ax.barh(y, counts, color=C_ESM3, edgecolor="white")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Enriched GO terms"); ax.set_title("Top ESM-3 features", fontsize=TITLE_SIZE)
    ax.invert_yaxis()
    panel_label(ax, "c")

    # Panel d: Top features ESM-2
    ax = axes[1, 1]
    pfr2 = go["esm2"].get("per_feature_results", [])
    top2 = sorted(pfr2, key=lambda f: -f.get("n_enriched_terms", len(f.get("enriched_terms", []))))[:20]
    names2 = [f"F{f.get('feature_id', i)}" for i, f in enumerate(top2)]
    counts2 = [f.get("n_enriched_terms", len(f.get("enriched_terms", []))) for f in top2]
    y2 = np.arange(len(names2))
    ax.barh(y2, counts2, color=C_ESM2, edgecolor="white")
    ax.set_yticks(y2); ax.set_yticklabels(names2, fontsize=6)
    ax.set_xlabel("Enriched GO terms"); ax.set_title("Top ESM-2 features", fontsize=TITLE_SIZE)
    ax.invert_yaxis()
    panel_label(ax, "d")

    save_fig(fig, "sfig05_go_enrichment", supplementary=True)

if __name__ == "__main__":
    generate()
