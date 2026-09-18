#!/usr/bin/env python3
"""SFig 7: Per-Type Probing & Feature Weights."""
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
    pt = load_json(RESULTS_SCALED / "per_feature_type_probing.json")
    ip = load_json(RESULTS_SCALED / "interpretable_probing.json")
    cm = load_json(RESULTS_SCALED / "cross_modal_features.json")

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4.5))
    fig.subplots_adjust(wspace=0.4)

    # Panel a: Per-annotation-type AUROC
    ax = axes[0]
    types_3 = pt["esm3"].get("per_type", {})
    types_2 = pt["esm2"].get("per_type", {})
    all_types = sorted(set(list(types_3.keys()) + list(types_2.keys())))
    # Filter to types with enough data
    valid = [t for t in all_types if types_3.get(t, {}).get("test_auroc", 0) > 0 or types_2.get(t, {}).get("test_auroc", 0) > 0]

    y = np.arange(len(valid))
    h = 0.35
    for i, t in enumerate(valid):
        auroc3 = types_3.get(t, {}).get("test_auroc", 0)
        auroc2 = types_2.get(t, {}).get("test_auroc", 0)
        if auroc3 > 0:
            ax.barh(i - h/2, auroc3, h, color=C_ESM3, edgecolor="white")
        if auroc2 > 0:
            ax.barh(i + h/2, auroc2, h, color=C_ESM2, edgecolor="white")

    ax.set_yticks(y); ax.set_yticklabels(valid, fontsize=6)
    ax.set_xlabel("AUROC")
    ax.axvline(0.5, color="grey", ls="--", lw=0.5)
    ax.legend(["ESM-3", "ESM-2"], fontsize=TICK_SIZE, loc="lower right")
    ax.set_title("AUROC by annotation type", fontsize=TITLE_SIZE)
    panel_label(ax, "a")

    # Panel b: Top probe features colored by cross-modal status
    ax = axes[1]
    enhanced_ids = set(cm.get("enhanced_feature_ids", []))
    suppressed_ids = set(cm.get("suppressed_feature_ids", []))

    top_feats = ip["esm3"].get("top_features", [])[:50]
    if top_feats:
        names = [f"F{f['feature_id']}" for f in top_feats]
        weights = [abs(f["weight"]) for f in top_feats]
        colors_bar = []
        for f in top_feats:
            fid = f["feature_id"]
            if fid in enhanced_ids:
                colors_bar.append(C_ENHANCED)
            elif fid in suppressed_ids:
                colors_bar.append(C_SUPPRESSED)
            else:
                colors_bar.append(C_INVARIANT)

        y = np.arange(len(names))
        ax.barh(y, weights, color=colors_bar, edgecolor="white")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("|Probe weight|")
        ax.invert_yaxis()

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=C_ENHANCED, label="Enhanced"),
            Patch(facecolor=C_SUPPRESSED, label="Suppressed"),
            Patch(facecolor=C_INVARIANT, label="Invariant"),
        ]
        ax.legend(handles=legend_elements, fontsize=6, loc="lower right")

    ax.set_title("Top ESM-3 probe features", fontsize=TITLE_SIZE)
    panel_label(ax, "b")

    save_fig(fig, "sfig07_probing_detail", supplementary=True)

if __name__ == "__main__":
    generate()
