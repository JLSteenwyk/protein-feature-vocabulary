#!/usr/bin/env python3
"""ED9: Feature Property Relationships, Co-activation & DMS Correlation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, matplotlib.pyplot as plt
from config import *; from style import setup_style; from utils import load_json, save_fig, panel_label
setup_style()

def generate():
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 5.5))
    fig.subplots_adjust(hspace=0.55, wspace=0.50)

    pca = load_json(RESULTS_SCALED / "decoder_pca.json")
    meta = pca["feature_metadata"]

    # Extract arrays
    conv = np.array([f["convergence_r"] for f in meta])
    n_go = np.array([f["n_go_terms"] for f in meta])
    cats = np.array([f["cross_modal_category"] for f in meta])

    enh_mask = cats == "enhanced"
    sup_mask = cats == "suppressed"
    inv_mask = cats == "invariant"
    colors_cat = [C_ENHANCED, C_SUPPRESSED, C_INVARIANT]
    positions = [0, 1, 2]

    # Panel a: Fraction GO-enriched by cross-modal category
    ax = axes[0, 0]
    frac_go = [
        (n_go[enh_mask] > 0).mean(),
        (n_go[sup_mask] > 0).mean(),
        (n_go[inv_mask] > 0).mean(),
    ]
    ax.bar(positions, frac_go, color=colors_cat, edgecolor="white")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Enh.", "Supp.", "Inv."], fontsize=TICK_SIZE)
    ax.set_ylabel("Fraction GO-enriched")
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(frac_go):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=6)
    panel_label(ax, "a")

    # Panel b: Median GO terms for enriched features by category
    ax = axes[0, 1]
    go_by_cat = {}
    for cat_name, mask in [("Enhanced", enh_mask), ("Suppressed", sup_mask), ("Invariant", inv_mask)]:
        go_vals = n_go[mask]
        go_enriched = go_vals[go_vals > 0]
        go_by_cat[cat_name] = {"n_enriched": len(go_enriched), "n_total": mask.sum(),
                                "median": float(np.median(go_enriched)) if len(go_enriched) > 0 else 0}

    medians = [go_by_cat[c]["median"] for c in ["Enhanced", "Suppressed", "Invariant"]]
    ax.bar(positions, medians, color=colors_cat, edgecolor="white")
    ax.set_xticks(positions)
    ax.set_xticklabels(["Enh.", "Supp.", "Inv."], fontsize=TICK_SIZE)
    ax.set_ylabel("Median GO terms\n(enriched features)")
    ax.set_ylim(bottom=0)
    for i, v in enumerate(medians):
        n_e = go_by_cat[["Enhanced", "Suppressed", "Invariant"][i]]["n_enriched"]
        n_t = go_by_cat[["Enhanced", "Suppressed", "Invariant"][i]]["n_total"]
        ax.text(i, v + max(medians) * 0.03, f"{v:.0f}\n({n_e}/{n_t})", ha="center", fontsize=6)
    panel_label(ax, "b")

    # Panel c: Cohen's d vs convergence scatter
    ax = axes[0, 2]
    cohens_d = np.array([f.get("cohens_d", 0) for f in meta])
    ax.scatter(conv[inv_mask], cohens_d[inv_mask], c=C_INVARIANT, s=1, alpha=0.08, rasterized=True)
    ax.scatter(conv[enh_mask], cohens_d[enh_mask], c=C_ENHANCED, s=3, alpha=0.4, rasterized=True)
    ax.scatter(conv[sup_mask], cohens_d[sup_mask], c=C_SUPPRESSED, s=3, alpha=0.4, rasterized=True)
    ax.axhline(0, color="grey", ls="--", lw=0.5)
    ax.axhline(0.2, color="grey", ls=":", lw=0.4)
    ax.axhline(-0.2, color="grey", ls=":", lw=0.4)
    ax.set_xlabel("Convergence (best-match r)")
    ax.set_ylabel("Cohen's d (S+St effect)")
    ax.set_xlim(0, 1)
    panel_label(ax, "c")

    # Panel d: empty (convergence violin promoted to Fig 2d)
    axes[0, 3].axis("off")

    # Row 2: Coactivation clusters + DMS
    co = load_json(RESULTS_SCALED / "coactivation_analysis.json")
    for col, (model, color, lab) in enumerate([("esm3", C_ESM3, "d"), ("esm2", C_ESM2, "e")]):
        ax = axes[1, col]
        try:
            clustering = co[model].get("clustering", {})
            cl = clustering.get("k10", clustering.get("clustering_k10", {}))
            if "clusters" in cl:
                clusters = cl["clusters"]
                sizes = [c["n_features"] for c in clusters]
                ax.bar(range(len(sizes)), sizes, color=color, edgecolor="white")
                ax.set_xlabel("Cluster"); ax.set_ylabel("Features")
                ax.set_ylim(bottom=0)
                for i, s in enumerate(sizes):
                    ax.text(i, s + max(sizes) * 0.02, str(s), ha="center",
                            fontsize=6, rotation=90)
            elif "cluster_sizes" in cl:
                sizes = cl["cluster_sizes"]
                ax.bar(range(len(sizes)), sizes, color=color, edgecolor="white")
                ax.set_xlabel("Cluster"); ax.set_ylabel("Features")
                ax.set_ylim(bottom=0)
            else:
                ax.text(0.5, 0.5, "No cluster data", transform=ax.transAxes, ha="center")
        except Exception:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
        panel_label(ax, lab)

    for col, (model, color, lab) in enumerate([("esm3", C_ESM3, "f"), ("esm2", C_ESM2, "g")]):
        ax = axes[1, col + 2]
        try:
            path = RESULTS_UNIFIED_ESM3 if "3" in model else RESULTS_UNIFIED_ESM2
            dm = load_json(path / "dms_correlation.json")
            agg = dm.get("aggregate", {})
            metrics = ["mutation_kl", "sae_activation", "mlm_entropy"]
            vals = [abs(agg.get(m, {}).get("mean_rho", agg.get(m, {}).get("mean_spearman", 0)))
                    for m in metrics]
            ax.bar(range(len(metrics)), vals, color=color, edgecolor="white")
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(["Mut. KL", "SAE act.", "MLM ent."],
                               fontsize=TICK_SIZE, rotation=30, ha="right")
            ax.set_ylabel("|mean rho|")
            ax.set_ylim(bottom=0)
            for i, v in enumerate(vals):
                ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=6)
        except Exception:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
        panel_label(ax, lab)

    save_fig(fig, "ed09_decoder_dms_phylo", supplementary=True)

if __name__ == "__main__":
    generate()
