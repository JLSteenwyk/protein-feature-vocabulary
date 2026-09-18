#!/usr/bin/env python3
"""ED8: Contact Prediction, OV/QK & Sparse Circuits (combines S12+S13)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, matplotlib.pyplot as plt
from config import *; from style import setup_style; from utils import load_json, save_fig, panel_label
setup_style()

def generate():
    ct=load_json(RESULTS_SCALED/"contact_map_from_attention.json")
    fig,axes=plt.subplots(2,3,figsize=(7.2,6.0)); fig.subplots_adjust(hspace=0.50,wspace=0.40)
    # Row 1: contact map
    ax=axes[0,0]; top=ct.get("top_heads",[])[:25]
    if top:
        names=[f"L{h['layer']}H{h['head']}" for h in top]
        precs=[h.get("precision_L",h.get("precision_at_L",0)) for h in top]
        colors_b=[BRIGHT[6] if h["layer"]>=32 else BRIGHT[1] if h["layer"]>=16 else BRIGHT[4] for h in top]
        ax.barh(range(len(names)),precs,color=colors_b,edgecolor="white")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("Precision @L"); ax.set_title("Top heads",fontsize=TITLE_SIZE); panel_label(ax,"a")

    ax=axes[0,1]; ps=ct.get("precision_summary",{})
    if ps:
        th=["L/5","L/2","L"]; x=np.arange(len(th)); w=0.35
        rv=[ps.get(t,{}).get("responsive",0) for t in th]; nv=[ps.get(t,{}).get("non_responsive",0) for t in th]
        ax.bar(x-w/2,rv,w,color=C_ENHANCED,label="Responsive",edgecolor="white")
        ax.bar(x+w/2,nv,w,color=C_S,label="Non-responsive",edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(th); ax.set_ylabel("Precision"); ax.legend(fontsize=TICK_SIZE)
    ax.set_title("Resp. vs non-resp.",fontsize=TITLE_SIZE); panel_label(ax,"b")

    ax=axes[0,2]; sup=ct.get("supervised_combination",{})
    tw=sup.get("top_weighted_heads",[])[:15]
    if tw:
        names=[f"L{h['layer']}H{h['head']}" for h in tw]; weights=[h["weight"] for h in tw]
        colors_w=[C_ENHANCED if h.get("is_responsive") else C_S for h in tw]
        ax.barh(range(len(names)),weights,color=colors_w,edgecolor="white")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names,fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("LR weight"); ax.set_title("Supervised weights",fontsize=TITLE_SIZE); panel_label(ax,"c")

    # Row 2: sparse circuits
    for col,(model,path,lab) in enumerate([("ESM-3",RESULTS_UNIFIED_ESM3/"sparse_feature_circuits.json","d"),
                                            ("ESM-2",RESULTS_UNIFIED_ESM2/"sparse_feature_circuits.json","e")]):
        ax=axes[1,col]
        try:
            sc=load_json(path); lps=sc.get("layer_pairs",{})
            names=[]; vals=[]
            for lp_name,lp_data in lps.items():
                conns=lp_data.get("connections",[])
                n_sig=sum(c.get("n_significant_upstream",0) for c in conns)
                names.append(lp_name); vals.append(n_sig)
            ax.bar(range(len(names)),vals,color=C_ESM3 if "3" in model else C_ESM2,edgecolor="white")
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names,fontsize=6,rotation=30,ha="right")
            ax.set_ylabel("Connections")
        except Exception: ax.text(0.5,0.5,"N/A",transform=ax.transAxes,ha="center")
        ax.set_title(f"{model} circuits",fontsize=TITLE_SIZE); panel_label(ax,lab)

    axes[1,2].axis("off")  # empty panel
    save_fig(fig,"ed08_contact_circuits",supplementary=True)

if __name__=="__main__": generate()
