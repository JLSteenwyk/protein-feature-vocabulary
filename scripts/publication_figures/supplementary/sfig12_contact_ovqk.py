#!/usr/bin/env python3
"""SFig 12: Contact Map & OV/QK Analysis."""
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
    ct = load_json(RESULTS_SCALED / "contact_map_from_attention.json")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # Panel a: Top heads by precision
    ax = axes[0, 0]
    top = ct.get("top_heads", [])[:30]
    if top:
        names = [f"L{h['layer']}H{h['head']}" for h in top]
        precs = [h.get("precision_L", h.get("precision_at_L", 0)) for h in top]
        y = np.arange(len(names))
        # Color by layer zone
        colors_bar = []
        for h in top:
            l = h["layer"]
            if l < 16: colors_bar.append(BRIGHT[4])  # light blue
            elif l < 32: colors_bar.append(BRIGHT[1])  # green
            else: colors_bar.append(BRIGHT[6])  # red-pink
        ax.barh(y, precs, color=colors_bar, edgecolor="white")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel("Contact precision @L")
        ax.invert_yaxis()
    ax.set_title("Top heads by precision", fontsize=TITLE_SIZE)
    panel_label(ax, "a")

    # Panel b: Precision summary (responsive vs non-responsive)
    ax = axes[0, 1]
    ps = ct.get("precision_summary", {})
    if ps:
        thresholds = ["L/5", "L/2", "L"]
        resp_vals = [ps.get(t, {}).get("responsive", 0) for t in thresholds]
        nonresp_vals = [ps.get(t, {}).get("non_responsive", 0) for t in thresholds]
        x = np.arange(len(thresholds))
        w = 0.35
        ax.bar(x - w/2, resp_vals, w, color=C_ENHANCED, label="Responsive", edgecolor="white")
        ax.bar(x + w/2, nonresp_vals, w, color=C_S, label="Non-responsive", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(thresholds)
        ax.set_ylabel("Mean precision"); ax.legend(fontsize=TICK_SIZE)
    ax.set_title("Responsive vs non-responsive", fontsize=TITLE_SIZE)
    panel_label(ax, "b")

    # Panel c: Supervised combination
    ax = axes[1, 0]
    sup = ct.get("supervised_combination", {})
    if sup:
        top_weights = sup.get("top_weighted_heads", [])[:20]
        if top_weights:
            names = [f"L{h['layer']}H{h['head']}" for h in top_weights]
            weights = [h["weight"] for h in top_weights]
            colors_w = [C_ENHANCED if h.get("is_responsive", False) else C_S for h in top_weights]
            y = np.arange(len(names))
            ax.barh(y, weights, color=colors_w, edgecolor="white")
            ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6)
            ax.set_xlabel("Logistic regression weight")
            ax.invert_yaxis()
        sup_prec = {k: v for k, v in sup.items() if k in ["L/5", "L/2", "L"]}
        if sup_prec:
            text = ", ".join(f"{k}={v:.3f}" for k, v in sup_prec.items())
            ax.set_title(f"Supervised weights ({text})", fontsize=6)
        else:
            ax.set_title("Supervised combination weights", fontsize=TITLE_SIZE)
    panel_label(ax, "c")

    # Panel d: OV/QK head classification
    ax = axes[1, 1]
    try:
        ovqk = load_json(RESULTS_UNIFIED_ESM3 / "ov_qk_decomposition.json")
        eval_layers = ovqk.get("eval_layers", [])
        if "per_layer" in ovqk:
            # Count head types per layer
            copy_counts, dist_counts, mixed_counts = [], [], []
            for l in eval_layers:
                ld = ovqk["per_layer"].get(str(l), {})
                heads = ld.get("heads", [])
                n_copy = sum(1 for h in heads if h.get("type") == "copy")
                n_dist = sum(1 for h in heads if h.get("type") == "distributed")
                n_mixed = len(heads) - n_copy - n_dist
                copy_counts.append(n_copy)
                dist_counts.append(n_dist)
                mixed_counts.append(n_mixed)
            x = np.arange(len(eval_layers))
            ax.bar(x, copy_counts, label="Copy", color=BRIGHT[0], edgecolor="white")
            ax.bar(x, dist_counts, bottom=copy_counts, label="Distributed", color=BRIGHT[1], edgecolor="white")
            ax.bar(x, mixed_counts, bottom=np.array(copy_counts)+np.array(dist_counts),
                   label="Mixed", color=BRIGHT[3], edgecolor="white")
            ax.set_xticks(x); ax.set_xticklabels(eval_layers, fontsize=6, rotation=45)
            ax.set_xlabel("Layer"); ax.set_ylabel("Head count")
            ax.legend(fontsize=TICK_SIZE)
        ax.set_title("OV/QK head types (ESM-3)", fontsize=TITLE_SIZE)
    except Exception:
        ax.text(0.5, 0.5, "OV/QK data unavailable", transform=ax.transAxes, ha="center")
    panel_label(ax, "d")

    save_fig(fig, "sfig12_contact_ovqk", supplementary=True)

if __name__ == "__main__":
    generate()
