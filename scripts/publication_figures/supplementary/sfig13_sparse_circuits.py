"""Supplementary Figure 13: Sparse feature circuits.

Layout: 1x3 panels (one per layer pair)
  For ESM-3 L16->L33 and L33->L42, plus ESM-2 L16->L24.
  Each panel shows top downstream features ranked by number of
  significant upstream connections (horizontal bar chart).
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
    esm3 = load_json(RESULTS_UNIFIED_ESM3 / "sparse_feature_circuits.json")
    esm2 = load_json(RESULTS_UNIFIED_ESM2 / "sparse_feature_circuits.json")
    return esm3, esm2


def _plot_panel(ax, connections, title, color, max_show=15):
    """Bar chart of downstream features by upstream connection count."""
    # Sort by n_significant_upstream descending
    sorted_conn = sorted(connections,
                         key=lambda c: c["n_significant_upstream"],
                         reverse=True)[:max_show]

    features = [str(c["downstream_feature"]) for c in sorted_conn]
    counts = [c["n_significant_upstream"] for c in sorted_conn]
    effects = [c["median_effect"] for c in sorted_conn]

    y_pos = np.arange(len(features))

    bars = ax.barh(y_pos, counts, color=color, edgecolor="white",
                   linewidth=0.3, height=0.7, alpha=0.85)

    # Annotate with median effect size
    for i, (cnt, eff) in enumerate(zip(counts, effects)):
        ax.text(cnt + 0.15, i, f"{eff:.1f}",
                va="center", ha="left", fontsize=6, color="0.3")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"F{f}" for f in features], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Significant upstream\nconnections")
    ax.set_title(title, fontsize=TITLE_SIZE)

    # Add a subtle annotation for the effect label
    ax.text(0.97, 0.02, "median effect",
            transform=ax.transAxes, fontsize=6, color="0.4",
            ha="right", va="bottom", style="italic")


def generate():
    esm3, esm2 = _load_data()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 6.0))
    fig.subplots_adjust(wspace=0.55, left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Panel a: ESM-3 L16 -> L33
    lp = esm3["layer_pairs"]["L16_to_L33"]
    _plot_panel(axes[0], lp["connections"],
                f"ESM-3: L16 \u2192 L33\n({lp['n_upstream_features']} up, "
                f"{len(lp['connections'])} down)",
                C_ESM3)
    panel_label(axes[0], "a", x=-0.22)

    # Panel b: ESM-3 L33 -> L42
    lp = esm3["layer_pairs"]["L33_to_L42"]
    _plot_panel(axes[1], lp["connections"],
                f"ESM-3: L33 \u2192 L42\n({lp['n_upstream_features']} up, "
                f"{len(lp['connections'])} down)",
                C_ESM3)
    panel_label(axes[1], "b", x=-0.22)

    # Panel c: ESM-2 L16 -> L24
    lp = esm2["layer_pairs"]["L16_to_L24"]
    _plot_panel(axes[2], lp["connections"],
                f"ESM-2: L16 \u2192 L24\n({lp['n_upstream_features']} up, "
                f"{len(lp['connections'])} down)",
                C_ESM2)
    panel_label(axes[2], "c", x=-0.22)

    fig.suptitle("Sparse feature circuits: upstream connectivity per downstream feature",
                 fontsize=TITLE_SIZE, y=0.98)

    save_fig(fig, "sfig13_sparse_circuits", supplementary=True)


if __name__ == "__main__":
    generate()
