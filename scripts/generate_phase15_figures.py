#!/usr/bin/env python3
"""Generate figures for Phase 1.5: Functional Site Attention.

Reads results from results/phase1/functional_attention/functional_attention_results.json
and the attention atlas head classifications.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY")
RESULTS_PATH = BASE / "results" / "phase1" / "functional_attention" / "functional_attention_results.json"
ATLAS_PATH = BASE / "results" / "phase1" / "attention_atlas" / "attention_atlas.json"
FIG_DIR = BASE / "results" / "figures" / "phase1"


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    with open(ATLAS_PATH) as f:
        atlas = json.load(f)

    head_classes = np.array(atlas["classifications"])  # (48, 24)
    target_layers = results["target_layers"]
    n_heads = results["n_heads"]
    n_proteins = results["n_proteins"]

    print(f"Phase 1.5 results: {n_proteins} proteins, {len(target_layers)} layers, {n_heads} heads")

    # Extract per-layer data
    by_layer = results["results_by_layer"]

    # ---- Figure 18: Enrichment heatmap (layers × heads) ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Build enrichment matrices
    enrich_s = np.zeros((len(target_layers), n_heads))
    enrich_st = np.zeros((len(target_layers), n_heads))

    for i, l in enumerate(target_layers):
        key = str(l)
        if key in by_layer:
            enrich_s[i] = by_layer[key]["enrichment_s"]
            enrich_st[i] = by_layer[key]["enrichment_st"]

    delta = enrich_st - enrich_s

    # Panel A: S+St enrichment
    ax = axes[0]
    im = ax.imshow(enrich_st, aspect='auto', cmap='RdYlBu_r', vmin=0.5, vmax=2.0)
    ax.set_yticks(range(len(target_layers)))
    ax.set_yticklabels([str(l) for l in target_layers])
    ax.set_xlabel('Head', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title('A. Functional Site Attention Enrichment (S+St)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Enrichment (func/nonfunc)', shrink=0.8)

    # Panel B: Delta (S+St - S)
    ax = axes[1]
    im = ax.imshow(delta, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax.set_yticks(range(len(target_layers)))
    ax.set_yticklabels([str(l) for l in target_layers])
    ax.set_xlabel('Head', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title('B. Structure-Induced Enrichment Change (S+St - S)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Δ Enrichment', shrink=0.8)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig18_functional_attention_heatmap.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Figure 19: Mean enrichment by layer, split by head type ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Mean enrichment across layers
    mean_s_by_layer = enrich_s.mean(axis=1)
    mean_st_by_layer = enrich_st.mean(axis=1)

    ax = axes[0]
    ax.plot(target_layers, mean_s_by_layer, 'o-', linewidth=2, color='#2196F3',
            label='S-only', markersize=7)
    ax.plot(target_layers, mean_st_by_layer, 's-', linewidth=2, color='#FF5722',
            label='S+St', markersize=7)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Functional Enrichment', fontsize=12)
    ax.set_title('A. Functional Attention Enrichment by Layer', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xticks(target_layers)
    ax.grid(alpha=0.3)

    # Panel B: Head type comparison at key layers
    ax = axes[1]
    key_layers = [l for l in [4, 6, 8, 32] if str(l) in by_layer]

    if key_layers:
        bar_data = []
        for l in key_layers:
            str_mask = head_classes[l] == "structure_responsive"
            seq_mask = head_classes[l] == "sequence_only"

            enr_st_arr = np.array(by_layer[str(l)]["enrichment_st"])
            enr_s_arr = np.array(by_layer[str(l)]["enrichment_s"])
            delta_arr = enr_st_arr - enr_s_arr

            sr_delta = delta_arr[str_mask].mean() if str_mask.any() else 0
            so_delta = delta_arr[seq_mask].mean() if seq_mask.any() else 0

            bar_data.append({
                "layer": l,
                "str_resp_delta": sr_delta,
                "seq_only_delta": so_delta,
                "n_str_resp": str_mask.sum(),
                "n_seq_only": seq_mask.sum(),
            })

        x = np.arange(len(bar_data))
        width = 0.35
        sr_vals = [d["str_resp_delta"] for d in bar_data]
        so_vals = [d["seq_only_delta"] for d in bar_data]

        ax.bar(x - width/2, sr_vals, width, label='Structure-responsive heads',
               color='#FF5722', alpha=0.85)
        ax.bar(x + width/2, so_vals, width, label='Sequence-only heads',
               color='#2196F3', alpha=0.85)
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Δ Enrichment (S+St - S)', fontsize=12)
        ax.set_title('B. Structure-Induced Change by Head Type', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(d["layer"]) for d in bar_data])
        ax.legend(fontsize=10)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig19_functional_attention_summary.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Figure 20: Per-head scatter (JSD vs functional enrichment delta) ----
    fig, ax = plt.subplots(figsize=(10, 7))

    # For each head in the structure-responsive zone, plot JSD vs functional delta
    jsd_all = []
    delta_all = []
    layer_all = []

    for l in target_layers:
        if str(l) not in by_layer or l >= 48:
            continue
        jsd_vals = atlas["jsd"][l]  # (24,)
        delta_vals = np.array(by_layer[str(l)]["enrichment_st"]) - np.array(by_layer[str(l)]["enrichment_s"])

        for h in range(n_heads):
            jsd_all.append(jsd_vals[h])
            delta_all.append(delta_vals[h])
            layer_all.append(l)

    jsd_all = np.array(jsd_all)
    delta_all = np.array(delta_all)
    layer_all = np.array(layer_all)

    # Color by layer zone
    colors = []
    for l in layer_all:
        if l <= 1:
            colors.append('#9E9E9E')
        elif l <= 9:
            colors.append('#FF5722')
        elif l <= 31:
            colors.append('#FF9800')
        else:
            colors.append('#2196F3')

    sc = ax.scatter(jsd_all, delta_all, c=colors, s=30, alpha=0.6)

    # Correlation
    valid = np.isfinite(jsd_all) & np.isfinite(delta_all)
    if valid.sum() > 2:
        r = np.corrcoef(jsd_all[valid], delta_all[valid])[0, 1]
        ax.text(0.95, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Structure Responsiveness (JSD)', fontsize=12)
    ax.set_ylabel('Functional Enrichment Change (S+St - S)', fontsize=12)
    ax.set_title('Structure Responsiveness vs Functional Site Attention',
                 fontsize=13, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='#9E9E9E', label='L0-1 (embedding)'),
        mpatches.Patch(facecolor='#FF5722', label='L2-9 (integration)'),
        mpatches.Patch(facecolor='#FF9800', label='L10-31 (refinement)'),
        mpatches.Patch(facecolor='#2196F3', label='L32-47 (unified)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='lower right')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig20_jsd_vs_functional.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved: fig18, fig19, fig20")


if __name__ == "__main__":
    main()
