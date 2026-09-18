#!/usr/bin/env python3
"""ED4: Feature Biology (combines S6+S7)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import Patch
from config import *; from style import setup_style; from utils import load_json, save_fig, panel_label
setup_style()

def generate():
    ra=load_json(RESULTS_SCALED/"residue_level_autointerpretability.json")
    pt=load_json(RESULTS_SCALED/"per_feature_type_probing.json")
    ip=load_json(RESULTS_SCALED/"interpretable_probing.json")
    cm=load_json(RESULTS_SCALED/"cross_modal_features.json")
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.0)); fig.subplots_adjust(hspace=0.45,wspace=0.40)
    # a: motif types
    ax=axes[0,0]; mtd=ra["motif_type_distribution"]; types=list(mtd.keys()); counts=[mtd[t] for t in types]
    ax.barh(range(len(types)),counts,color=[BRIGHT[i%len(BRIGHT)] for i in range(len(types))],edgecolor="white")
    ax.set_yticks(range(len(types))); ax.set_yticklabels(types,fontsize=6); ax.set_xlabel("Features"); ax.invert_yaxis()
    panel_label(ax,"a")
    # b: AA composition of a representative feature
    ax=axes[0,1]; feat=next((f for f in ra["per_feature"] if f.get("top_amino_acids")),None)
    if feat:
        aa=feat["top_amino_acids"]; aas=[a["aa"] for a in aa[:15]]; fracs=[a["count"] for a in aa[:15]]
        ax.bar(range(len(aas)),fracs,color=BRIGHT[0],edgecolor="white"); ax.set_xticks(range(len(aas))); ax.set_xticklabels(aas,fontsize=6)
        ax.set_ylabel("Count"); ax.set_xlabel("Amino acid")
    panel_label(ax,"b")
    # c: per-type AUROC
    ax=axes[1,0]; t3=pt["esm3"].get("per_type",{}); t2=pt["esm2"].get("per_type",{})
    valid=[t for t in sorted(set(list(t3.keys())+list(t2.keys()))) if t3.get(t,{}).get("test_auroc",0)>0 or t2.get(t,{}).get("test_auroc",0)>0]
    h=0.35
    for i,t in enumerate(valid):
        a3=t3.get(t,{}).get("test_auroc",0); a2=t2.get(t,{}).get("test_auroc",0)
        if a3>0: ax.barh(i-h/2,a3,h,color=C_ESM3,edgecolor="white")
        if a2>0: ax.barh(i+h/2,a2,h,color=C_ESM2,edgecolor="white")
    ax.set_yticks(range(len(valid))); ax.set_yticklabels(valid,fontsize=6); ax.set_xlabel("AUROC")
    ax.axvline(0.5,color="grey",ls="--",lw=0.5)
    panel_label(ax,"c")
    # d: top probe features
    ax=axes[1,1]; enh=set(cm.get("enhanced_feature_ids",[])); sup=set(cm.get("suppressed_feature_ids",[]))
    top=ip["esm3"].get("top_features",[])[:25]  # Show 25 instead of 50 for readability
    if top:
        names=[f"F{f['feature_id']}" for f in top]; weights=[abs(f["weight"]) for f in top]
        colors_f=[C_ENHANCED if f["feature_id"] in enh else C_SUPPRESSED if f["feature_id"] in sup else C_INVARIANT for f in top]
        y=np.arange(len(names))
        ax.barh(y,weights,height=0.7,color=colors_f,edgecolor="white",linewidth=0.3)
        ax.set_yticks(y); ax.set_yticklabels(names,fontsize=6)
        ax.set_xlabel("|Weight|"); ax.invert_yaxis()
    panel_label(ax,"d")
    save_fig(fig,"ed04_feature_biology",supplementary=True)

if __name__=="__main__": generate()
