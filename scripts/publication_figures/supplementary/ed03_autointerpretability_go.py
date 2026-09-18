#!/usr/bin/env python3
"""ED3: Autointerpretability & GO Enrichment (combines S4+S5)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, matplotlib.pyplot as plt
from config import *; from style import setup_style; from utils import load_json, save_fig, panel_label
setup_style()

def generate():
    ai=load_json(RESULTS_SCALED/"autointerpretability.json")
    go=load_json(RESULTS_SCALED/"go_enrichment.json")
    fig,axes=plt.subplots(2,4,figsize=(7.2,5.5)); fig.subplots_adjust(hspace=0.45,wspace=0.40)
    # Row 1: autointerp histograms
    for col,(lab,model,key,title,color) in enumerate([
        ("a","esm3","claude_pearson_r","ESM-3/Claude",C_ESM3),("b","esm3","gpt_pearson_r","ESM-3/GPT",C_ESM3),
        ("c","esm2","claude_pearson_r","ESM-2/Claude",C_ESM2),("d","esm2","gpt_pearson_r","ESM-2/GPT",C_ESM2)]):
        ax=axes[0,col]; pf=ai[model]["per_feature"]
        vals=np.array([f.get(key) for f in pf if f.get(key) is not None and not(isinstance(f.get(key),float) and np.isnan(f.get(key)))],dtype=float)
        ax.hist(vals,bins=30,color=color,alpha=0.8,edgecolor="white")
        med=np.median(vals); ax.axvline(med,color="black",ls="--",lw=1)
        ax.text(0.03,0.95,f"n={len(vals)}/{len(pf)}\nmed={med:.3f}",transform=ax.transAxes,ha="left",va="top",fontsize=6)
        ax.set_xlabel("Pearson r"); ax.set_ylabel("Count"); ax.set_xlim(-0.5,1.0); panel_label(ax,lab)
    # Row 2: GO enrichment
    for col,(model,color,lab) in enumerate([("esm3",C_ESM3,"e"),("esm2",C_ESM2,"f")]):
        ax=axes[1,col]; pfr=go[model].get("per_feature_results",[])
        nt=np.array([f.get("n_enriched_terms",0) for f in pfr])
        if (nt>0).any(): ax.hist(nt[nt>0],bins=30,color=color,alpha=0.8,edgecolor="white")
        med=np.median(nt[nt>0]) if (nt>0).any() else 0; ax.axvline(med,color="black",ls="--",lw=1)
        frac=go[model].get("frac_enriched",(nt>0).mean())
        ax.text(0.97,0.95,f"{frac*100:.1f}% enriched\nmed={med:.0f}",transform=ax.transAxes,ha="right",va="top",fontsize=6)
        ax.set_xlabel("GO terms/feature"); ax.set_ylabel("Count"); panel_label(ax,lab)
    for col,(model,color,lab) in enumerate([("esm3",C_ESM3,"g"),("esm2",C_ESM2,"h")]):
        ax=axes[1,col+2]; pfr=go[model].get("per_feature_results",[])
        top=sorted(pfr,key=lambda f:-f.get("n_enriched_terms",0))[:20]
        names=[f"F{f.get('feature_id',i)}" for i,f in enumerate(top)]
        counts=[f.get("n_enriched_terms",0) for f in top]; y=np.arange(len(names))
        ax.barh(y,counts,color=color,edgecolor="white"); ax.set_yticks(y); ax.set_yticklabels(names,fontsize=6)
        ax.set_xlabel("GO terms"); ax.invert_yaxis(); panel_label(ax,lab)
    save_fig(fig,"ed02_autointerpretability_go",supplementary=True)

if __name__=="__main__": generate()
