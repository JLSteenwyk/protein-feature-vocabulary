#!/usr/bin/env python3
"""SFig 9: Logit Lens, Tuned Lens & Residual Decomposition."""
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
    ll = load_json(RESULTS_SCALED / "logit_lens.json")
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, 6.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # Logit lens layers and per-layer data
    layers = sorted(ll["per_layer"].keys(), key=int)
    layer_nums = [int(l) for l in layers]

    # Panel a: Logit lens accuracy S vs S+St
    ax = axes[0, 0]
    acc_s = [ll["per_layer"][l]["accuracy_S"] for l in layers]
    acc_sst = [ll["per_layer"][l]["accuracy_SSt"] for l in layers]
    ax.plot(layer_nums, acc_s, "-o", color=C_S, markersize=2, label="S-only")
    ax.plot(layer_nums, acc_sst, "-s", color=C_SST, markersize=2, label="S+St")
    ax.set_xlabel("Layer"); ax.set_ylabel("Next-token accuracy")
    ax.legend(fontsize=6, frameon=False); 
    panel_label(ax, "a")

    # Panel b: Logit lens KL
    ax = axes[0, 1]
    kl_s = [ll["per_layer"][l]["kl_S"] for l in layers]
    kl_sst = [ll["per_layer"][l]["kl_SSt"] for l in layers]
    ax.plot(layer_nums, kl_s, "-o", color=C_S, markersize=2, label="S-only")
    ax.plot(layer_nums, kl_sst, "-s", color=C_SST, markersize=2, label="S+St")
    ax.set_xlabel("Layer"); ax.set_ylabel("KL from final logits")
    ax.legend(fontsize=6, frameon=False); 
    panel_label(ax, "b")

    # Panel c: Tuned lens vs raw (ESM-3)
    ax = axes[0, 2]
    try:
        tl = load_json(RESULTS_UNIFIED_ESM3 / "tuned_lens.json")
        eval_layers = tl.get("eval_layers", [])
        if "per_layer" in tl:
            def _get_val(d, key):
                v = d.get(key, 0)
                return v.get("mean", v) if isinstance(v, dict) else v
            raw_acc = [_get_val(tl["per_layer"][str(l)], "raw_top1") for l in eval_layers]
            tuned_acc = [_get_val(tl["per_layer"][str(l)], "tuned_top1") for l in eval_layers]
            ax.plot(eval_layers, raw_acc, "-o", color=C_S, markersize=2, label="Raw")
            ax.plot(eval_layers, tuned_acc, "-s", color=C_SST, markersize=2, label="Tuned")
            ax.legend(fontsize=6, frameon=False)
        ax.set_xlabel("Layer"); ax.set_ylabel("Top-1 accuracy")
    except Exception:
        ax.text(0.5, 0.5, "Tuned lens\ndata unavailable", transform=ax.transAxes, ha="center")
    panel_label(ax, "c")

    # Panel d: Accuracy difference (S+St - S)
    ax = axes[1, 0]
    acc_diff = [ll["per_layer"][l].get("accuracy_diff", acc_sst[i] - acc_s[i]) for i, l in enumerate(layers)]
    ax.bar(layer_nums, acc_diff, color=C_SST, width=1.5, edgecolor="white")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Accuracy diff (S+St - S)")
    
    panel_label(ax, "d")

    # Panel e: Residual decomposition ESM-3
    ax = axes[1, 1]
    try:
        rd3 = load_json(RESULTS_UNIFIED_ESM3 / "residual_decomposition.json")
        rd_layers = sorted(rd3["per_layer"].keys(), key=int)
        rd_lnums = [int(l) for l in rd_layers]
        attn_norm = [rd3["per_layer"][l].get("attn_l2_norm", rd3["per_layer"][l].get("mean_attn_norm", 0)) for l in rd_layers]
        mlp_norm = [rd3["per_layer"][l].get("mlp_l2_norm", rd3["per_layer"][l].get("mean_mlp_norm", 0)) for l in rd_layers]
        ax.plot(rd_lnums, attn_norm, "-", color=C_ESM3, label="Attn", linewidth=1.2)
        ax.plot(rd_lnums, mlp_norm, "--", color=C_ESM2, label="MLP", linewidth=1.2)
        ax.legend(fontsize=6, frameon=False)
        ax.set_xlabel("Layer"); ax.set_ylabel("L2 norm")
        
    except Exception:
        ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes, ha="center")
    panel_label(ax, "e")

    # Panel f: Residual decomposition ESM-2
    ax = axes[1, 2]
    try:
        rd2 = load_json(RESULTS_UNIFIED_ESM2 / "residual_decomposition.json")
        rd_layers = sorted(rd2["per_layer"].keys(), key=int)
        rd_lnums = [int(l) for l in rd_layers]
        attn_norm = [rd2["per_layer"][l].get("attn_l2_norm", rd2["per_layer"][l].get("mean_attn_norm", 0)) for l in rd_layers]
        mlp_norm = [rd2["per_layer"][l].get("mlp_l2_norm", rd2["per_layer"][l].get("mean_mlp_norm", 0)) for l in rd_layers]
        ax.plot(rd_lnums, attn_norm, "-", color=C_ESM3, label="Attn", linewidth=1.2)
        ax.plot(rd_lnums, mlp_norm, "--", color=C_ESM2, label="MLP", linewidth=1.2)
        ax.legend(fontsize=6, frameon=False)
        ax.set_xlabel("Layer"); ax.set_ylabel("L2 norm")
        
    except Exception:
        ax.text(0.5, 0.5, "Data unavailable", transform=ax.transAxes, ha="center")
    panel_label(ax, "f")

    save_fig(fig, "sfig09_lens_decomposition", supplementary=True)

if __name__ == "__main__":
    generate()
