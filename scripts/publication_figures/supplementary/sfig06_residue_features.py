#!/usr/bin/env python3
"""SFig 6: Residue-Level Feature Examples."""
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
    ra = load_json(RESULTS_SCALED / "residue_level_autointerpretability.json")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 5.0))
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # Panel a: Motif type distribution
    ax = axes[0, 0]
    mtd = ra["motif_type_distribution"]
    types = list(mtd.keys())
    counts = [mtd[t] for t in types]
    colors_bar = [BRIGHT[i % len(BRIGHT)] for i in range(len(types))]
    ax.barh(range(len(types)), counts, color=colors_bar, edgecolor="white")
    ax.set_yticks(range(len(types))); ax.set_yticklabels(types, fontsize=6)
    ax.set_xlabel("Number of features")
    ax.set_title("Motif type distribution", fontsize=TITLE_SIZE)
    ax.invert_yaxis()
    panel_label(ax, "a")

    # Panels b-d: Examples from different motif types
    pf = ra["per_feature"]
    type_examples = {}
    for f in pf:
        mt = f.get("motif_type", "unknown")
        if mt not in type_examples:
            type_examples[mt] = f

    example_types = ["secondary_structure", "sequence_pattern", "functional_site"]
    panel_labs = ["b", "c", "d"]

    for idx, (mt, lab) in enumerate(zip(example_types, panel_labs)):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        if mt in type_examples:
            feat = type_examples[mt]
            desc = feat.get("description", mt)[:60]
            aa_comp = feat.get("top_amino_acids", [])
            if aa_comp:
                if isinstance(aa_comp, list):
                    aas = [a["aa"] for a in aa_comp[:15]]
                    fracs = [a["count"] for a in aa_comp[:15]]
                else:
                    aas = list(aa_comp.keys())[:15]
                    fracs = [aa_comp[a] for a in aas]
                ax.bar(range(len(aas)), fracs, color=BRIGHT[idx], edgecolor="white")
                ax.set_xticks(range(len(aas))); ax.set_xticklabels(aas, fontsize=6)
                ax.set_ylabel("Fraction")
            ax.set_title(f"{mt}\n{desc}", fontsize=6)
        else:
            ax.text(0.5, 0.5, f"No {mt} example", transform=ax.transAxes, ha="center")
        panel_label(ax, lab)

    save_fig(fig, "sfig06_residue_features", supplementary=True)

if __name__ == "__main__":
    generate()
