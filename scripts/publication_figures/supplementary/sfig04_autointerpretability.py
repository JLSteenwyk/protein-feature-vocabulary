#!/usr/bin/env python3
"""SFig 4: Autointerpretability Distributions."""
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
    ai = load_json(RESULTS_SCALED / "autointerpretability.json")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    panels = [
        ("a", "esm3", "claude_pearson_r", "ESM-3 / Claude", C_ESM3),
        ("b", "esm3", "gpt_pearson_r", "ESM-3 / GPT-4o", C_ESM3),
        ("c", "esm2", "claude_pearson_r", "ESM-2 / Claude", C_ESM2),
        ("d", "esm2", "gpt_pearson_r", "ESM-2 / GPT-4o", C_ESM2),
    ]

    for idx, (lab, model, key, title, color) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        pf = ai[model]["per_feature"]
        vals = [f.get(key) for f in pf if f.get(key) is not None and not (isinstance(f.get(key), float) and np.isnan(f.get(key)))]
        vals = np.array(vals, dtype=float)
        n_valid = len(vals)
        n_total = len(pf)

        ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor="white")
        med = np.median(vals)
        ax.axvline(med, color="black", ls="--", lw=1)
        ax.text(0.95, 0.95, f"n={n_valid}/{n_total}\nmedian={med:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_xlabel("Pearson r")
        ax.set_ylabel("Count")
        ax.set_title(title, fontsize=TITLE_SIZE)
        ax.set_xlim(-0.5, 1.0)
        panel_label(ax, lab)

    save_fig(fig, "sfig04_autointerpretability", supplementary=True)

if __name__ == "__main__":
    generate()
