#!/usr/bin/env python3
"""ED1: SAE Training & Reconstruction Quality (combines S1+S2)."""
import sys
from pathlib import Path
_src_dir = str(Path(__file__).parent.parent.parent.parent / "src")
_fig_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, _src_dir)
sys.path.insert(0, _fig_dir)

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from config import *
from style import setup_style
from utils import load_json, save_fig, panel_label
setup_style()

def generate():
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.50,
                           left=0.08, right=0.97, top=0.97, bottom=0.05)

    # Row 1: Training curves (split into two sub-panels) + cosine + dead
    # Panel a: ESM-3 training loss
    ax = fig.add_subplot(gs[0, 0])
    p = MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt"
    if p.exists():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        lh = ck.get("log_history", [])
        ax.plot([e["step"] for e in lh], [e["loss"] for e in lh], color=C_ESM3, lw=1.2)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("ESM-3", fontsize=TICK_SIZE)
    panel_label(ax, "a")

    # Panel b: ESM-2 training loss (own y-axis scale)
    ax = fig.add_subplot(gs[0, 1])
    p = MODEL_ROOT / "esm2_residue_ef8_k64" / "best.pt"
    if p.exists():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        lh = ck.get("log_history", [])
        ax.plot([e["step"] for e in lh], [e["loss"] for e in lh], color=C_ESM2, lw=1.2)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title("ESM-2", fontsize=TICK_SIZE)
    panel_label(ax, "b")

    # Panel c: Cosine similarity across configs
    ax = fig.add_subplot(gs[0, 2])
    xp = 0; xt, xl = [], []
    for m, c in [("esm3", C_ESM3), ("esm2", C_ESM2)]:
        for e in load_json(MODEL_ROOT / f"{m}_sae_training_summary.json"):
            ax.bar(xp, e["cosine"], color=c, width=0.7, edgecolor="white")
            xt.append(xp); xl.append(e["tag"].replace(f"{m}_", "")); xp += 1
        xp += 0.5
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Cosine"); ax.set_ylim(0, 1.05)
    panel_label(ax, "c")

    # Panel d: Dead neuron fraction
    ax = fig.add_subplot(gs[0, 3])
    xp = 0; xt, xl = [], []
    for m, c in [("esm3", C_ESM3), ("esm2", C_ESM2)]:
        for e in load_json(MODEL_ROOT / f"{m}_sae_training_summary.json"):
            ax.bar(xp, e["dead_frac"] * 100, color=c, width=0.7, edgecolor="white")
            xt.append(xp); xl.append(e["tag"].replace(f"{m}_", "")); xp += 1
        xp += 0.5
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Dead (%)"); ax.set_ylim(0, 105)
    panel_label(ax, "d")

    # Row 2: Sparsity + Reconstruction (cosine, variance, MSE)
    # Panel e: L0 sparsity
    ax = fig.add_subplot(gs[1, 0])
    sp = load_json(RESULTS_SCALED / "feature_sparsity_analysis.json")
    for m, c in [("esm3", C_ESM3), ("esm2", C_ESM2)]:
        h = sp[m]["residue"]["l0_distribution"]["histogram"]
        cn = np.array(h["counts"]); ed = np.array(h["bin_edges"])
        ct = (ed[:-1] + ed[1:]) / 2
        ax.step(ct, cn, color=c, label=m.upper(), lw=1.2, where="mid")
    ax.set_xlabel("L0"); ax.set_ylabel("Count")
    panel_label(ax, "e")

    # Reconstruction data
    rq = load_json(RESULTS_SCALED / "reconstruction_quality.json")
    labs, cv, cs, vv, vs, mv, ms = [], [], [], [], [], [], []
    for m in ["esm3", "esm2"]:
        for lv in ["residue", "protein"]:
            if lv in rq[m]:
                r = rq[m][lv]
                labs.append(f"{m[:4].upper()}\n{lv.capitalize()}")
                cv.append(r["cosine_similarity"]["mean"]); cs.append(r["cosine_similarity"]["std"])
                vv.append(r["variance_explained"]["mean"]); vs.append(r["variance_explained"]["std"])
                mv.append(r["mse"]["mean"]); ms.append(r["mse"]["std"])
    cc = [C_ESM3, C_ESM3, C_ESM2, C_ESM2]
    x = np.arange(len(labs))

    # Panel f: Cosine similarity by level
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(x, cv, yerr=cs, color=cc, capsize=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("Cosine"); ax.set_ylim(0, 1.05)
    panel_label(ax, "f")

    # Panel g: Variance explained
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(x, vv, yerr=vs, color=cc, capsize=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("Var. explained"); ax.set_ylim(0, 1.05)
    panel_label(ax, "g")

    # Panel h: MSE (log scale)
    ax = fig.add_subplot(gs[1, 3])
    ax.bar(x, mv, yerr=ms, color=cc, capsize=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=6)
    ax.set_ylabel("MSE"); ax.set_yscale("log")
    panel_label(ax, "h")

    # Row 3: Percentile distribution (wider)
    ax = fig.add_subplot(gs[2, :2])
    pct = ["5", "25", "50", "75", "95"]
    for m, c, mk in [("esm3", C_ESM3, "o"), ("esm2", C_ESM2, "s")]:
        if "residue" in rq[m]:
            p = rq[m]["residue"]["cosine_similarity"]["percentiles"]
            ax.plot(pct, [p[k] for k in pct], f"-{mk}", color=c, markersize=5,
                    lw=1.2, label=m.upper())
    ax.set_xlabel("Percentile"); ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=TICK_SIZE, frameon=False)
    panel_label(ax, "i")

    # Hide unused bottom-right panels
    fig.add_subplot(gs[2, 2]).axis("off")
    fig.add_subplot(gs[2, 3]).axis("off")

    save_fig(fig, "ed01_training_reconstruction", supplementary=True)

if __name__ == "__main__": generate()
