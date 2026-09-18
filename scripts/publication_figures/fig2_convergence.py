#!/usr/bin/env python3
"""Figure 2: ESM-2 and ESM-3 converge on shared biological representations.

Layout: 2x2 (a-d)
  a — Best-match correlation histogram (ESM-3->ESM-2 vs permutation null)
  b — Fraction above thresholds (grouped bar, both directions + null)
  c — UMAP of decoder weights colored by convergence
  d — Evolutionary correlation (SAE magnitude vs MLM entropy) for both models

Usage:
    ./env/bin/python scripts/publication_figures/generate_all.py --fig 2
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label


def _load_convergence():
    """Load feature convergence data."""
    return load_json(RESULTS_SCALED / "feature_convergence.json")


def _load_evolutionary():
    """Load evolutionary correlation data for both models."""
    esm3 = load_json(RESULTS_UNIFIED_ESM3 / "evolutionary_correlation.json")
    esm2 = load_json(RESULTS_UNIFIED_ESM2 / "evolutionary_correlation.json")
    return esm3, esm2


def _panel_a(ax, conv):
    """Best-match correlation histogram: real vs null."""
    # Real distribution: ESM-3 -> ESM-2 best-match Pearson r
    r_values = np.array(conv["esm3_to_esm2"]["pearson"]["r_distribution"]["values"])
    median_r = conv["esm3_to_esm2"]["pearson"]["median_best_r"]
    frac_above_03 = conv["esm3_to_esm2"]["pearson"]["frac_above_thresholds"]["0.3"]

    # Null summary statistics
    null = conv["null_distribution"]
    null_median = null["median_best_r"]
    null_frac_03 = null["frac_above_thresholds"]["0.3"]

    # Histogram
    bins = np.linspace(0, 1, 51)
    ax.hist(r_values, bins=bins, color=C_ESM3, alpha=0.8, edgecolor='white',
            linewidth=0.3, zorder=3)

    # Vertical dashed lines with labels
    ax.axvline(median_r, color=C_ESM3, linestyle='--', linewidth=0.8, zorder=4)
    ax.axvline(0.3, color='#555555', linestyle=':', linewidth=0.6, zorder=4)

    ymax = ax.get_ylim()[1]
    ax.text(median_r + 0.02, ymax * 0.98, f"Median\n({median_r:.2f})",
            fontsize=TICK_SIZE, color=C_ESM3, va='top')
    ax.text(0.28, ymax * 0.98, "r = 0.3",
            fontsize=TICK_SIZE, color='#555555', va='top', ha='right')

    ax.set_xlabel("Best-match Pearson r")
    ax.set_ylabel("Number of features")
    ax.set_xlim(0, 1)


def _panel_b(ax, conv):
    """Fraction above thresholds — grouped bar chart."""
    thresholds = ["0.3", "0.5", "0.7", "0.9"]
    x_labels = [f"r > {t}" for t in thresholds]

    # Extract fractions for each direction
    esm3_to_esm2 = conv["esm3_to_esm2"]["pearson"]["frac_above_thresholds"]
    esm2_to_esm3 = conv["esm2_to_esm3"]["pearson"]["frac_above_thresholds"]
    null = conv["null_distribution"]["frac_above_thresholds"]

    vals_32 = [esm3_to_esm2[t] for t in thresholds]
    vals_23 = [esm2_to_esm3[t] for t in thresholds]
    vals_null = [null[t] for t in thresholds]

    x = np.arange(len(thresholds))
    width = 0.25

    bars1 = ax.bar(x - width, vals_32, width, color=C_ESM3, edgecolor='white',
                   linewidth=0.3, label="ESM-3 \u2192 ESM-2", zorder=3)
    bars2 = ax.bar(x, vals_23, width, color=C_ESM2, edgecolor='white',
                   linewidth=0.3, label="ESM-2 \u2192 ESM-3", zorder=3)
    bars3 = ax.bar(x + width, vals_null, width, color=C_INVARIANT, edgecolor='white',
                   linewidth=0.3, label="Null", zorder=3)

    # Value labels on top of bars — small font, rotated to avoid overlap
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.001:
                label = f"{h:.0%}" if h >= 0.10 else f"{h:.1%}"
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        label, ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Fraction of features")
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', frameon=False, fontsize=6)


def _panel_c(ax):
    """Convergence by cross-modal category (violin plot)."""
    pca_data = load_json(RESULTS_SCALED / "decoder_pca.json")
    meta = pca_data["feature_metadata"]

    conv = np.array([f["convergence_r"] for f in meta])
    cats = np.array([f["cross_modal_category"] for f in meta])

    enh_mask = cats == "enhanced"
    sup_mask = cats == "suppressed"
    inv_mask = cats == "invariant"

    data = [conv[enh_mask], conv[sup_mask], conv[inv_mask]]
    labels = ["Enh.", "Supp.", "Inv."]
    colors = [C_ENHANCED, C_SUPPRESSED, C_INVARIANT]
    positions = [0, 1, 2]

    parts = ax.violinplot(data, positions=positions, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)

    # Median lines with values
    for i, d in enumerate(data):
        med = np.median(d)
        ax.hlines(med, positions[i] - 0.3, positions[i] + 0.3, color="black", lw=1.2)
        ax.text(positions[i] + 0.35, med, f"{med:.3f}", fontsize=6, va="center")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Convergence (best-match r)")
    ax.set_ylim(0, 1.05)


def _panel_d(ax, evo_esm3, evo_esm2):
    """Evolutionary correlation bar chart: SAE magnitude vs MLM entropy."""
    # Aggregate Spearman rho (entropy_vs_sae)
    categories = ["buried", "exposed", "functional", "nonfunctional"]
    cat_labels = ["Buried", "Exposed", "Functional", "Non-func."]

    esm3_vals = [evo_esm3["breakdown"][c]["entropy_vs_sae_spearman"] for c in categories]
    esm2_vals = [evo_esm2["breakdown"][c]["entropy_vs_sae_spearman"] for c in categories]

    x = np.arange(len(categories))
    width = 0.35

    bars3 = ax.bar(x - width / 2, esm3_vals, width, color=C_ESM3,
           edgecolor='white', linewidth=0.3, label="ESM-3", zorder=3)
    bars2 = ax.bar(x + width / 2, esm2_vals, width, color=C_ESM2,
           edgecolor='white', linewidth=0.3, label="ESM-2", zorder=3)

    # Value labels below bars (negative values)
    for bars, vals in [(bars3, esm3_vals), (bars2, esm2_vals)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v - 0.02,
                    f"{v:.2f}", ha="center", va="top", fontsize=6, rotation=90, color="#333333")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, rotation=30, ha='right')
    ax.set_ylabel("Spearman ρ")
    ax.set_ylim(-0.7, 0.05)
    ax.legend(loc='lower center', frameon=False, fontsize=6, ncol=2)

    # Aggregate median text removed — goes in figure legend


def _panel_e(ax):
    """Convergent vs unique feature probing AUROC."""
    aa = load_json(RESULTS_SCALED / "additional_analyses.json")
    cu = aa["convergent_vs_unique_probing"]

    categories = ["All\n(12,288)", "Convergent\n(3,163)", "Unique\n(638)"]
    aurocs = [cu["all_features"]["auroc"], cu["convergent_only"]["auroc"], cu["unique_only"]["auroc"]]
    colors_bar = [C_ESM3, BRIGHT[1], BRIGHT[3]]  # dark blue, green, grey

    bars = ax.bar(range(3), aurocs, color=colors_bar, edgecolor="white", linewidth=0.3, zorder=3)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, aurocs)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    ax.axhline(0.5, color="grey", ls="--", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories, fontsize=6)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # Title removed — goes in figure legend


def generate():
    """Generate Figure 2."""
    setup_style()

    # Load data
    conv = _load_convergence()
    evo_esm3, evo_esm2 = _load_evolutionary()

    # Create figure — Row 1: a, b (wide). Row 2: c, d, e (with more spacing)
    fig = plt.figure(figsize=(DOUBLE_COL, 5.5))
    import matplotlib.gridspec as gridspec

    # Use separate GridSpecs for each row to control spacing independently
    gs_top = gridspec.GridSpec(1, 2, figure=fig, hspace=0.0, wspace=0.35,
                               left=0.08, right=0.97, top=0.96, bottom=0.56)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig, hspace=0.0, wspace=0.55,
                               left=0.08, right=0.97, top=0.46, bottom=0.06)

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_d = fig.add_subplot(gs_bot[0, 1])
    ax_e = fig.add_subplot(gs_bot[0, 2])

    # Draw panels
    _panel_a(ax_a, conv)
    _panel_b(ax_b, conv)

    # Panel c: probing AUROC
    _panel_e(ax_c)

    # Panel d: GO term overlap
    aa = load_json(RESULTS_SCALED / "additional_analyses.json")
    go_ov = aa["go_term_overlap"]
    go_categories = ["ESM-3 terms\nin ESM-2", "ESM-2 terms\nin ESM-3", "Jaccard\nindex"]
    vals = [go_ov["esm3_per_feature_overlap"], go_ov["esm2_per_feature_overlap"], go_ov["jaccard_index"]]
    colors_f = [C_ESM3, C_ESM2, BRIGHT[5]]
    ax_d.bar(range(3), vals, color=colors_f, edgecolor="white", linewidth=0.3)
    for i, v in enumerate(vals):
        ax_d.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=6)
    ax_d.set_xticks(range(3))
    ax_d.set_xticklabels(go_categories, fontsize=6)
    ax_d.set_ylabel("GO term overlap\n(fraction)")
    ax_d.set_ylim(0, 1.1)

    # Panel e: evolutionary correlation (was d)
    _panel_d(ax_e, evo_esm3, evo_esm2)

    # Panel labels — offset to avoid overlapping figure content
    panel_label(ax_a, 'a', x=-0.06)
    panel_label(ax_b, 'b', x=-0.08)
    panel_label(ax_c, 'c', x=-0.10)
    panel_label(ax_d, 'd', x=-0.10)
    panel_label(ax_e, 'e', x=-0.10)

    save_fig(fig, "fig2_convergence")


if __name__ == "__main__":
    generate()
