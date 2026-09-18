"""Supplementary Figure 15: DMS correlation and phylogenetic recapitulation.

Layout: 2x2 panels
  a — DMS Spearman rho per protein for ESM-3 (grouped bars: mutation_kl,
      sae_activation, mlm_entropy)
  b — Same for ESM-2
  c — Phylogenetic recapitulation: mean Mantel rho by layer for ESM-3
  d — Same for ESM-2
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label

setup_style()

import matplotlib.pyplot as plt
import numpy as np


def _load_data():
    dms_esm3 = load_json(RESULTS_UNIFIED_ESM3 / "dms_correlation.json")
    dms_esm2 = load_json(RESULTS_UNIFIED_ESM2 / "dms_correlation.json")
    phylo_esm3 = load_json(RESULTS_UNIFIED_ESM3 / "phylogenetic_recapitulation.json")
    phylo_esm2 = load_json(RESULTS_UNIFIED_ESM2 / "phylogenetic_recapitulation.json")
    return dms_esm3, dms_esm2, phylo_esm3, phylo_esm2


# Short display names for DMS proteins
_PROTEIN_SHORT = {
    "GFP_Sarkisyan": "GFP",
    "TEM1_Stiffler": "TEM-1",
    "GAL4_Kitzman": "GAL4",
    "PTEN_Mighell": "PTEN",
    "BRCA1_Findlay": "BRCA1",
    "SPIKE_Starr_bind": "SPIKE",
    "P53_Giacomelli": "P53",
}

# Metric display names and colors
_METRICS = [
    ("mutation_kl", "Mutation KL", BRIGHT[0]),    # blue
    ("sae_activation", "SAE activation", BRIGHT[6]),  # red-pink
    ("mlm_entropy", "MLM entropy", BRIGHT[1]),    # green
]


def _plot_dms_panel(ax, dms_data, title):
    """Grouped bar chart of Spearman rho per protein per metric."""
    proteins = list(dms_data["per_protein"].keys())
    n_proteins = len(proteins)
    n_metrics = len(_METRICS)

    bar_width = 0.22
    x = np.arange(n_proteins)

    for i, (metric_key, metric_label, color) in enumerate(_METRICS):
        rhos = []
        for prot in proteins:
            rho = dms_data["per_protein"][prot]["correlations"][metric_key]["spearman_rho"]
            rhos.append(rho)
        offset = (i - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(x + offset, rhos, width=bar_width, color=color,
                      alpha=0.8, label=metric_label, edgecolor="white",
                      linewidth=0.3)

    ax.axhline(0, color="0.5", linewidth=0.4, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([_PROTEIN_SHORT.get(p, p) for p in proteins],
                       rotation=35, ha="right", fontsize=6)
    ax.set_ylabel("Spearman \u03c1")
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE - 0.5, frameon=False, loc="upper left",
              ncol=1, handlelength=1.2, handletextpad=0.4)
    ax.set_ylim(-0.7, 0.6)


def _plot_phylo_panel(ax, phylo_data, title, color):
    """Line plot of mean Mantel rho by layer."""
    agg = phylo_data["aggregate_by_layer"]
    layers = sorted(agg.keys(), key=lambda k: int(k))
    layer_ints = [int(l) for l in layers]

    means = [agg[l]["mean_mantel_rho"] for l in layers]
    stds = [agg[l]["std_mantel_rho"] for l in layers]

    means = np.array(means)
    stds = np.array(stds)

    ax.fill_between(layer_ints, means - stds, means + stds,
                    color=color, alpha=0.15)
    ax.plot(layer_ints, means, "-o", color=color, markersize=3,
            linewidth=1.0, zorder=3)

    # Mark n_significant
    for i, l in enumerate(layers):
        n_sig = agg[l]["n_significant"]
        n_fam = agg[l]["n_families"]
        frac = n_sig / n_fam
        if frac < 0.7:
            ax.annotate(f"{n_sig}/{n_fam}",
                        (layer_ints[i], means[i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=6, ha="center", color="0.4")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Mantel \u03c1")
    ax.set_title(title, fontsize=TITLE_SIZE)
    ax.set_ylim(0.3, 0.65)


def generate():
    dms_esm3, dms_esm2, phylo_esm3, phylo_esm2 = _load_data()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    fig.subplots_adjust(hspace=0.55, wspace=0.35, left=0.08, right=0.96,
                        top=0.93, bottom=0.10)

    # Panel a: DMS ESM-3
    _plot_dms_panel(axes[0, 0], dms_esm3,
                    f"DMS correlation (ESM-3, L{dms_esm3['sae_layer']})")
    panel_label(axes[0, 0], "a")

    # Panel b: DMS ESM-2
    _plot_dms_panel(axes[0, 1], dms_esm2,
                    f"DMS correlation (ESM-2, L{dms_esm2['sae_layer']})")
    panel_label(axes[0, 1], "b")

    # Panel c: Phylogenetic ESM-3
    _plot_phylo_panel(axes[1, 0], phylo_esm3,
                      f"Phylogenetic recapitulation (ESM-3, {phylo_esm3['n_families']} families)",
                      C_ESM3)
    panel_label(axes[1, 0], "c")

    # Panel d: Phylogenetic ESM-2
    _plot_phylo_panel(axes[1, 1], phylo_esm2,
                      f"Phylogenetic recapitulation (ESM-2, {phylo_esm2['n_families']} families)",
                      C_ESM2)
    panel_label(axes[1, 1], "d")

    save_fig(fig, "sfig15_dms_phylogenetic", supplementary=True)


if __name__ == "__main__":
    generate()
