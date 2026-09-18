"""Supplementary Figure 16: Full case-study protein SAE activation heatmaps.

Layout: 5 vertical heatmap strips (one per protein), each showing the
top 10 features. Below each heatmap, a binary annotation row marks
functional site positions (estimated from high-activation overlap).
Uses viridis colormap with a single shared colorbar.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label

setup_style()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import numpy as np


def _load_data():
    return load_json(RESULTS_SCALED / "case_study_proteins.json")


def _build_heatmap(case_study, n_features=10):
    """Build a (n_features, length) activation matrix from active_positions."""
    length = case_study["length"]
    features = case_study["top_features"][:n_features]
    mat = np.zeros((n_features, length), dtype=float)
    labels = []

    for i, feat in enumerate(features):
        act = feat["mean_activation"]
        for pos in feat["active_positions"]:
            if pos < length:
                mat[i, pos] = act
        modal = feat.get("cross_modal_label", "")
        labels.append(f"F{feat['feature_id']}")

    return mat, labels


def generate():
    data = _load_data()
    all_case_studies = data["case_studies"]
    # Keep only Serralysin (idx 2), cytochrome c (idx 1), OmcB (idx 3)
    # Reorder: Serralysin first (clearest feature-site alignment)
    case_studies = [all_case_studies[2], all_case_studies[1], all_case_studies[3]]
    n_cases = len(case_studies)
    n_features = 10

    fig = plt.figure(figsize=(7.2, 7.0))

    # Use GridSpec: n_cases rows, 2 cols (heatmap + tiny colorbar col)
    gs = gridspec.GridSpec(n_cases, 2, width_ratios=[1, 0.025],
                           hspace=0.45, wspace=0.05,
                           left=0.10, right=0.88, top=0.96, bottom=0.04)

    panel_labels = ["a", "b", "c"]
    vmin, vmax = 0, 0

    # First pass: find global vmax
    all_mats = []
    all_labels = []
    for cs in case_studies:
        mat, labels = _build_heatmap(cs, n_features)
        all_mats.append(mat)
        all_labels.append(labels)
        vmax = max(vmax, mat.max())

    # Second pass: plot
    for idx, (cs, mat, feat_labels) in enumerate(zip(case_studies, all_mats,
                                                      all_labels)):
        ax = fig.add_subplot(gs[idx, 0])

        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                        vmin=vmin, vmax=vmax, interpolation="nearest")

        # Y-axis: feature labels
        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(feat_labels, fontsize=6)

        # X-axis: residue positions (sparse ticks)
        length = cs["length"]
        step = max(1, length // 8)
        xticks = list(range(0, length, step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks], fontsize=6)

        # Title: protein name + accession
        name_short = cs["name"][:35]
        title = f"{cs['accession']} - {name_short} (L={length})"
        ax.set_title(title, fontsize=TITLE_SIZE - 1, pad=3)

        if idx == n_cases - 1:
            ax.set_xlabel("Residue position")

        panel_label(ax, panel_labels[idx], x=-0.08, y=1.12)

        # Functional site markers (red triangles below heatmap)
        try:
            meta = load_json(RESULTS_SCALED.parent.parent / "data" / "eval_expanded" / "metadata.json")
            accession = cs["accession"]
            if accession in meta:
                func_sites = meta[accession].get("features", [])
                site_positions = []
                for fs in func_sites:
                    ft = fs.get("type", "")
                    if ft in ["Active site", "Binding site", "Metal binding",
                              "Site", "Disulfide bond", "Modified residue"]:
                        ps = fs.get("start", 1) - 1
                        pe = fs.get("end", ps + 1)
                        for p in range(ps, pe):
                            if 0 <= p < length:
                                site_positions.append(p)
                if site_positions:
                    ax.scatter(site_positions, [n_features - 0.3] * len(site_positions),
                               marker="^", s=8, color="#CC3311", zorder=10,
                               clip_on=False, linewidths=0)
        except Exception:
            pass

    # Shared colorbar — height of panel a (index 0, Serralysin)
    cbar_ax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label("Mean activation", fontsize=AXIS_LABEL_SIZE)
    cb.ax.tick_params(labelsize=6)

    save_fig(fig, "sfig16_case_studies_full", supplementary=True)


if __name__ == "__main__":
    generate()
