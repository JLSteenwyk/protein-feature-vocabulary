#!/usr/bin/env python3
"""
Generate all Phase 1 publication-quality figures from saved results.

Figures:
  1. CKA Modality Integration Map
  2. SAE Feature Comparison
  3. Attention Atlas Heatmap
  4. JSD by Layer
  5. Entropy vs JSD Scatter
  6. DSSP SAE Structural Enrichment
  7. Top Structure-Responsive Heads Profile
"""

import matplotlib
matplotlib.use('Agg')

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')

# ── Global style settings ────────────────────────────────────────────────────
LABEL_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

# Consistent color palette
COLOR_S = '#2171b5'        # Blue for S-only
COLOR_SST = '#e6550d'      # Orange/red for S+St
COLOR_SF = '#31a354'       # Green for S+F
COLOR_SALL = '#756bb1'     # Purple for S+St+F
COLOR_EARLY = '#d62728'    # Red for early layers
COLOR_MID = '#ff7f0e'      # Orange for mid layers
COLOR_LATE = '#1f77b4'     # Blue for late layers

plt.rcParams.update({
    'font.size': TICK_SIZE,
    'axes.labelsize': LABEL_SIZE,
    'axes.titlesize': TITLE_SIZE,
    'legend.fontsize': LEGEND_SIZE,
    'xtick.labelsize': TICK_SIZE,
    'ytick.labelsize': TICK_SIZE,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE = Path('/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY')
RESULTS = BASE / 'results' / 'phase1'
OUTDIR = BASE / 'results' / 'figures' / 'phase1'
OUTDIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    """Save figure as both PNG (300 dpi) and PDF."""
    png_path = OUTDIR / f'{name}.png'
    pdf_path = OUTDIR / f'{name}.pdf'
    fig.savefig(png_path, dpi=300, facecolor='white', edgecolor='none')
    fig.savefig(pdf_path, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  Saved: {png_path}')
    print(f'  Saved: {pdf_path}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: CKA Modality Integration Map
# ═══════════════════════════════════════════════════════════════════════════════
def figure1_cka_modality():
    print('\n[Figure 1] CKA Modality Integration Map')

    with open(RESULTS / 'modality_dropout' / 'cka_results.json') as f:
        data = json.load(f)

    layers = data['layers']
    conditions = data['conditions']  # S, S+St, S+F, S+St+F
    # Condition indices: 0=S, 1=S+St, 2=S+F, 3=S+St+F

    cka_s_sst = []     # CKA(S, S+St)
    cka_s_sf = []      # CKA(S, S+F)
    cka_sst_sstf = []  # CKA(S+St, S+St+F)

    for layer in layers:
        mat = data['cka_matrices'][str(layer)]
        cka_s_sst.append(mat[0][1])
        cka_s_sf.append(mat[0][2])
        cka_sst_sstf.append(mat[1][3])

    cka_s_sst = np.array(cka_s_sst)
    cka_s_sf = np.array(cka_s_sf)
    cka_sst_sstf = np.array(cka_sst_sstf)

    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    # Primary y-axis: CKA similarity
    l1, = ax1.plot(layers, cka_s_sst, 'o-', color=COLOR_SST, linewidth=2,
                   markersize=6, label='CKA(S, S+St)', zorder=3)
    l2, = ax1.plot(layers, cka_s_sf, 's-', color=COLOR_SF, linewidth=2,
                   markersize=6, label='CKA(S, S+F)', zorder=3)
    l3, = ax1.plot(layers, cka_sst_sstf, '^-', color=COLOR_SALL, linewidth=2,
                   markersize=6, label='CKA(S+St, S+St+F)', zorder=3)

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('CKA Similarity')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(layers[0] - 1, layers[-1] + 1)

    # Secondary y-axis: 1-CKA (modality impact) as inset
    ax_inset = fig.add_axes([0.18, 0.55, 0.30, 0.32])  # [left, bottom, width, height]
    ax_inset.fill_between(layers, 1 - cka_s_sst, alpha=0.3, color=COLOR_SST)
    ax_inset.fill_between(layers, 1 - cka_s_sf, alpha=0.3, color=COLOR_SF)
    ax_inset.plot(layers, 1 - cka_s_sst, '-', color=COLOR_SST, linewidth=1.5,
                  label='Structure impact')
    ax_inset.plot(layers, 1 - cka_s_sf, '-', color=COLOR_SF, linewidth=1.5,
                  label='Function impact')
    ax_inset.set_xlabel('Layer', fontsize=9)
    ax_inset.set_ylabel('1 - CKA\n(Modality Impact)', fontsize=9)
    ax_inset.set_title('Modality Impact', fontsize=10, fontweight='bold')
    ax_inset.tick_params(labelsize=8)
    ax_inset.legend(fontsize=7, loc='upper right')
    ax_inset.set_ylim(-0.05, 1.05)
    ax_inset.set_xlim(layers[0], layers[-1])
    ax_inset.grid(True, alpha=0.3)

    # Add annotations
    # Find where CKA(S, S+F) crosses 0.5
    for i in range(len(layers) - 1):
        if cka_s_sf[i] < 0.5 and cka_s_sf[i+1] >= 0.5:
            cross_layer = layers[i] + (layers[i+1] - layers[i]) * (0.5 - cka_s_sf[i]) / (cka_s_sf[i+1] - cka_s_sf[i])
            ax1.axvline(cross_layer, color='gray', linestyle='--', alpha=0.5, zorder=1)
            ax1.annotate(f'Function integrates\n~L{int(cross_layer)}',
                        xy=(cross_layer, 0.5), fontsize=9,
                        xytext=(cross_layer + 4, 0.35),
                        arrowprops=dict(arrowstyle='->', color='gray'),
                        color='gray')
            break

    # Note the high CKA(S+St, S+St+F) throughout
    ax1.annotate('S+St and S+St+F remain\nhighly similar (>0.99)',
                xy=(16, cka_sst_sstf[4]), fontsize=9,
                xytext=(22, 0.8),
                arrowprops=dict(arrowstyle='->', color=COLOR_SALL),
                color=COLOR_SALL)

    ax1.legend(handles=[l1, l2, l3], loc='center right', framealpha=0.9)
    ax1.set_title('Modality Integration Across ESM-3 Layers', fontweight='bold')

    save_fig(fig, 'fig1_cka_modality')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: SAE Feature Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def figure2_sae_features():
    print('\n[Figure 2] SAE Feature Comparison')

    with open(RESULTS / 'modality_saes' / 'cross_condition_comparison.json') as f:
        data = json.load(f)

    target_layers = [16, 33, 42]
    x = np.arange(len(target_layers))
    width = 0.35

    # Extract data
    s_aa = [data[str(l)]['s_only']['strong_aa_features'] for l in target_layers]
    sst_aa = [data[str(l)]['s_plus_st']['strong_aa_features'] for l in target_layers]
    s_func = [data[str(l)]['s_only']['functional_features_2x'] for l in target_layers]
    sst_func = [data[str(l)]['s_plus_st']['functional_features_2x'] for l in target_layers]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5))

    # Subplot A: AA-enriched features (>3x)
    bars1 = ax_a.bar(x - width/2, s_aa, width, color=COLOR_S, label='S-only',
                     edgecolor='white', linewidth=0.8, zorder=3)
    bars2 = ax_a.bar(x + width/2, sst_aa, width, color=COLOR_SST, label='S+St',
                     edgecolor='white', linewidth=0.8, zorder=3)

    ax_a.set_xlabel('Layer')
    ax_a.set_ylabel('Number of Features')
    ax_a.set_title('A. AA-Enriched Features (>3x)', fontweight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f'L{l}' for l in target_layers])
    ax_a.legend()

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                 f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                 f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add fold-change annotations
    for i, l in enumerate(target_layers):
        if sst_aa[i] > 0:
            fold = s_aa[i] / sst_aa[i]
            ax_a.text(x[i], max(s_aa[i], sst_aa[i]) + 5,
                     f'{fold:.1f}x', ha='center', fontsize=9, color='#333333',
                     fontstyle='italic')

    # Subplot B: Functional site features (>2x)
    bars3 = ax_b.bar(x - width/2, s_func, width, color=COLOR_S, label='S-only',
                     edgecolor='white', linewidth=0.8, zorder=3)
    bars4 = ax_b.bar(x + width/2, sst_func, width, color=COLOR_SST, label='S+St',
                     edgecolor='white', linewidth=0.8, zorder=3)

    ax_b.set_xlabel('Layer')
    ax_b.set_ylabel('Number of Features')
    ax_b.set_title('B. Functional Site Features (>2x)', fontweight='bold')
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f'L{l}' for l in target_layers])
    ax_b.legend()

    # Add value labels
    for bar in bars3:
        h = bar.get_height()
        if h > 0:
            ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                     f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars4:
        h = bar.get_height()
        if h > 0:
            ax_b.text(bar.get_x() + bar.get_width()/2., h + 0.3,
                     f'{int(h)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle('Modality-Specific SAE Features Across Layers', fontsize=TITLE_SIZE,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig2_sae_features')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Attention Atlas Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def figure3_attention_atlas():
    print('\n[Figure 3] Attention Atlas Heatmap')

    with open(RESULTS / 'attention_atlas' / 'attention_atlas.json') as f:
        data = json.load(f)

    jsd = np.array(data['jsd'])  # 48 x 24
    n_layers, n_heads = jsd.shape

    fig, ax = plt.subplots(figsize=(10, 12))

    # Use 'hot_r' so high JSD is dark/hot and low is white
    im = ax.imshow(jsd, aspect='auto', cmap='inferno',
                   interpolation='nearest', vmin=0, vmax=np.percentile(jsd, 99))

    ax.set_xlabel('Head Index')
    ax.set_ylabel('Layer')
    ax.set_title('Structure Responsiveness Per Attention Head', fontweight='bold')

    # X ticks
    ax.set_xticks(np.arange(0, n_heads, 2))
    ax.set_xticklabels(np.arange(0, n_heads, 2))

    # Y ticks - show every 4th layer
    ax.set_yticks(np.arange(0, n_layers, 4))
    ax.set_yticklabels(np.arange(0, n_layers, 4))

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label('Jensen-Shannon Divergence', fontsize=LABEL_SIZE)

    # Zone annotations on right side
    # Structure-responsive zone: L2-9 (early layers with high JSD)
    y_sr_start, y_sr_end = 2, 9
    ax.annotate('', xy=(n_heads + 0.8, y_sr_start), xytext=(n_heads + 0.8, y_sr_end),
               arrowprops=dict(arrowstyle='<->', color=COLOR_EARLY, lw=2))
    ax.text(n_heads + 1.3, (y_sr_start + y_sr_end) / 2,
            'Structure-\nresponsive\nzone\n(L2-9)',
            fontsize=9, color=COLOR_EARLY, fontweight='bold',
            va='center', ha='left')

    # Transition zone: L10-31
    y_tr_start, y_tr_end = 10, 31
    ax.annotate('', xy=(n_heads + 0.8, y_tr_start), xytext=(n_heads + 0.8, y_tr_end),
               arrowprops=dict(arrowstyle='<->', color=COLOR_MID, lw=2))
    ax.text(n_heads + 1.3, (y_tr_start + y_tr_end) / 2,
            'Transition\nzone\n(L10-31)',
            fontsize=9, color=COLOR_MID, fontweight='bold',
            va='center', ha='left')

    # Sequence-only zone: L32-47
    y_sq_start, y_sq_end = 32, 47
    ax.annotate('', xy=(n_heads + 0.8, y_sq_start), xytext=(n_heads + 0.8, y_sq_end),
               arrowprops=dict(arrowstyle='<->', color=COLOR_LATE, lw=2))
    ax.text(n_heads + 1.3, (y_sq_start + y_sq_end) / 2,
            'Sequence-\nonly zone\n(L32-47)',
            fontsize=9, color=COLOR_LATE, fontweight='bold',
            va='center', ha='left')

    # Expand xlim to make room for annotations
    ax.set_xlim(-0.5, n_heads + 5)

    fig.tight_layout()
    save_fig(fig, 'fig3_attention_atlas')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: JSD by Layer
# ═══════════════════════════════════════════════════════════════════════════════
def figure4_jsd_by_layer():
    print('\n[Figure 4] JSD by Layer')

    with open(RESULTS / 'attention_atlas' / 'attention_atlas.json') as f:
        atlas = json.load(f)
    with open(RESULTS / 'attention_atlas' / 'deep_analysis_data.json') as f:
        deep = json.load(f)

    jsd_matrix = np.array(atlas['jsd'])  # 48 x 24
    n_layers, n_heads = jsd_matrix.shape

    lt = deep['layer_transition']
    mean_jsd = np.array(lt['mean_jsd_per_layer'])
    std_jsd = np.array(lt['std_jsd_per_layer'])
    peak_layer = lt['peak_layer']
    transition_layer = lt['transition_layer_median']
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Shading for +/- 1 std
    ax.fill_between(layers, mean_jsd - std_jsd, mean_jsd + std_jsd,
                    alpha=0.2, color=COLOR_S, zorder=1)

    # Individual head dots (transparent)
    for head in range(n_heads):
        ax.scatter(layers, jsd_matrix[:, head], s=8, alpha=0.12,
                  color='gray', zorder=2, rasterized=True)

    # Mean line
    ax.plot(layers, mean_jsd, '-', color=COLOR_S, linewidth=2.5,
            label='Mean JSD', zorder=4)

    # Sigmoid fit
    fitted = np.array(lt['sigmoid_fit']['fitted_values'])
    ax.plot(layers, fitted, '--', color='#e377c2', linewidth=1.5,
            label=f'Sigmoid fit (R$^2$={lt["sigmoid_fit"]["r_squared"]:.2f})',
            zorder=3, alpha=0.8)

    # Mark peak layer
    ax.annotate(f'Peak: L{peak_layer}\n(JSD={mean_jsd[peak_layer]:.3f})',
               xy=(peak_layer, mean_jsd[peak_layer]),
               xytext=(peak_layer + 5, mean_jsd[peak_layer] + 0.04),
               fontsize=10, fontweight='bold', color=COLOR_EARLY,
               arrowprops=dict(arrowstyle='->', color=COLOR_EARLY, lw=1.5),
               zorder=5)

    # Mark transition point
    ax.axvline(transition_layer, color='gray', linestyle=':', alpha=0.6, zorder=1)
    ax.annotate(f'Transition: ~L{transition_layer}',
               xy=(transition_layer, mean_jsd[transition_layer]),
               xytext=(transition_layer + 4, mean_jsd[transition_layer] + 0.03),
               fontsize=10, color='gray',
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
               zorder=5)

    # Color background zones
    ax.axvspan(-0.5, 9.5, alpha=0.05, color=COLOR_EARLY, zorder=0)
    ax.axvspan(9.5, 31.5, alpha=0.03, color=COLOR_MID, zorder=0)
    ax.axvspan(31.5, 47.5, alpha=0.05, color=COLOR_LATE, zorder=0)

    # Zone labels at top
    ax.text(4.5, ax.get_ylim()[1] * 0.95, 'Early', fontsize=10,
            color=COLOR_EARLY, ha='center', fontweight='bold', alpha=0.7)
    ax.text(20.5, ax.get_ylim()[1] * 0.95, 'Middle', fontsize=10,
            color=COLOR_MID, ha='center', fontweight='bold', alpha=0.7)
    ax.text(39.5, ax.get_ylim()[1] * 0.95, 'Late', fontsize=10,
            color=COLOR_LATE, ha='center', fontweight='bold', alpha=0.7)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Jensen-Shannon Divergence')
    ax.set_title('Attention Structure-Responsiveness Across Layers', fontweight='bold')
    ax.set_xlim(-0.5, 47.5)
    ax.set_ylim(bottom=-0.01)
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, 'fig4_jsd_by_layer')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Entropy vs JSD Scatter
# ═══════════════════════════════════════════════════════════════════════════════
def figure5_entropy_jsd():
    print('\n[Figure 5] Entropy vs JSD Scatter')

    with open(RESULTS / 'attention_atlas' / 'deep_analysis_data.json') as f:
        deep = json.load(f)

    evj = deep['entropy_vs_jsd']
    jsd = np.array(evj['scatter_jsd'])
    entropy_delta = np.array(evj['scatter_entropy_delta'])
    classification = evj['scatter_classification']

    # Assign layer index to each data point (1152 = 48 layers * 24 heads)
    n_layers = 48
    n_heads = 24
    layer_idx = np.repeat(np.arange(n_layers), n_heads)

    # Color by layer group
    colors = np.empty(len(jsd), dtype=object)
    for i in range(len(jsd)):
        l = layer_idx[i]
        if l < 10:
            colors[i] = COLOR_EARLY
        elif l < 32:
            colors[i] = COLOR_MID
        else:
            colors[i] = COLOR_LATE

    fig, ax = plt.subplots(figsize=(9, 7))

    # Scatter by group for legend
    mask_early = layer_idx < 10
    mask_mid = (layer_idx >= 10) & (layer_idx < 32)
    mask_late = layer_idx >= 32

    ax.scatter(jsd[mask_late], entropy_delta[mask_late], s=12, alpha=0.25,
              color=COLOR_LATE, label='Late (L32-47)', zorder=2, rasterized=True)
    ax.scatter(jsd[mask_mid], entropy_delta[mask_mid], s=12, alpha=0.25,
              color=COLOR_MID, label='Middle (L10-31)', zorder=3, rasterized=True)
    ax.scatter(jsd[mask_early], entropy_delta[mask_early], s=15, alpha=0.4,
              color=COLOR_EARLY, label='Early (L0-9)', zorder=4, rasterized=True)

    # Regression line
    # Use robust fit on all data
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(jsd, entropy_delta, 1)
    x_fit = np.linspace(0, jsd.max(), 100)
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, 'k--', linewidth=1.5, alpha=0.7, zorder=5,
            label=f'Linear fit (r={evj["pearson_r"]:.2f})')

    # Reference line at y=0
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3, zorder=1)

    # Annotate
    ax.annotate('High JSD heads show\nentropy decrease\n(attention focuses)',
               xy=(0.45, -2.5), fontsize=10, fontstyle='italic',
               color='#555555',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Jensen-Shannon Divergence (S vs S+St)')
    ax.set_ylabel('Entropy Change (S+St minus S)')
    ax.set_title('Structure-Responsive Heads Focus Attention (Entropy Decreases)',
                fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, 'fig5_entropy_jsd')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: DSSP SAE Structural Enrichment
# ═══════════════════════════════════════════════════════════════════════════════
def figure6_dssp_sae():
    print('\n[Figure 6] DSSP SAE Structural Enrichment')

    target_layers = [16, 33, 42]
    conditions = [('S', 'S-only'), ('S_St', 'S+St')]

    data_dict = {}
    for cond_key, cond_label in conditions:
        for layer in target_layers:
            fn = RESULTS / 'modality_saes' / 'structural_annotations' / f'{cond_key}_layer_{layer}_structural.json'
            with open(fn) as f:
                d = json.load(f)
            data_dict[(cond_key, layer)] = d['summary']

    x = np.arange(len(target_layers))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel A: Fraction of features with strong SS enrichment (>2x) ──
    s_frac = [data_dict[('S', l)]['strong_ss_enrichment_fraction'] for l in target_layers]
    sst_frac = [data_dict[('S_St', l)]['strong_ss_enrichment_fraction'] for l in target_layers]

    bars1 = ax1.bar(x - width/2, [f * 100 for f in s_frac], width,
                    color=COLOR_S, label='S-only', edgecolor='white',
                    linewidth=0.8, zorder=3)
    bars2 = ax1.bar(x + width/2, [f * 100 for f in sst_frac], width,
                    color=COLOR_SST, label='S+St', edgecolor='white',
                    linewidth=0.8, zorder=3)

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_xlabel('Layer')
    ax1.set_ylabel('% Features with Strong SS Enrichment')
    ax1.set_title('A. Strong SS Enrichment (>2x)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'L{l}' for l in target_layers])
    ax1.legend()

    # ── Panel B: Mean max SS enrichment ──
    s_mean = [data_dict[('S', l)]['mean_max_ss_enrichment'] for l in target_layers]
    sst_mean = [data_dict[('S_St', l)]['mean_max_ss_enrichment'] for l in target_layers]

    bars3 = ax2.bar(x - width/2, s_mean, width, color=COLOR_S, label='S-only',
                    edgecolor='white', linewidth=0.8, zorder=3)
    bars4 = ax2.bar(x + width/2, sst_mean, width, color=COLOR_SST, label='S+St',
                    edgecolor='white', linewidth=0.8, zorder=3)

    for bar in bars3:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar in bars4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Reference line at 1.0 (no enrichment)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax2.text(x[-1] + 0.55, 1.02, 'No enrichment', fontsize=8, color='gray')

    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Mean Max SS Enrichment')
    ax2.set_title('B. Mean Max Secondary Structure Enrichment', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'L{l}' for l in target_layers])
    ax2.legend()

    fig.suptitle('S-only SAEs Learn More Structural Features Than S+St',
                fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig6_dssp_sae')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Top Structure-Responsive Heads Profile
# ═══════════════════════════════════════════════════════════════════════════════
def figure7_top_heads():
    print('\n[Figure 7] Top Structure-Responsive Heads Profile')

    with open(RESULTS / 'attention_atlas' / 'deep_analysis_data.json') as f:
        deep = json.load(f)

    profiles = deep['top10_profiles']['profiles']

    n_heads = len(profiles)
    head_labels = [f"L{p['layer']}H{p['head']}" for p in profiles]

    # Prepare data for heatmap-style profile
    jsd_vals = [p['jsd'] for p in profiles]
    entropy_s = [p['entropy_s'] for p in profiles]
    entropy_st = [p['entropy_st'] for p in profiles]
    entropy_delta = [p['entropy_delta'] for p in profiles]
    local_s = [p['local_fraction_s'] for p in profiles]
    local_st = [p['local_fraction_st'] for p in profiles]
    max_attn_s = [p['mean_max_attn_s'] for p in profiles]
    max_attn_st = [p['mean_max_attn_st'] for p in profiles]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [1.5, 2, 2]})

    # ── Panel A: Horizontal bar chart of JSD ──
    ax_bar = axes[0]
    y_pos = np.arange(n_heads)
    bars = ax_bar.barh(y_pos, jsd_vals, color=[COLOR_EARLY if profiles[i]['layer'] < 10
                       else COLOR_MID for i in range(n_heads)],
                       edgecolor='white', linewidth=0.5, zorder=3)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(head_labels, fontsize=11, fontweight='bold')
    ax_bar.set_xlabel('JSD')
    ax_bar.set_title('A. JSD Score', fontweight='bold')
    ax_bar.invert_yaxis()

    # Add JSD values
    for i, v in enumerate(jsd_vals):
        ax_bar.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

    # ── Panel B: Entropy comparison (S vs S+St) ──
    ax_ent = axes[1]
    bar_h = 0.35
    ax_ent.barh(y_pos - bar_h/2, entropy_s, bar_h, color=COLOR_S,
                label='Entropy (S)', edgecolor='white', linewidth=0.5, zorder=3)
    ax_ent.barh(y_pos + bar_h/2, entropy_st, bar_h, color=COLOR_SST,
                label='Entropy (S+St)', edgecolor='white', linewidth=0.5, zorder=3)

    ax_ent.set_yticks(y_pos)
    ax_ent.set_yticklabels([])
    ax_ent.set_xlabel('Attention Entropy (bits)')
    ax_ent.set_title('B. Entropy: S vs S+St', fontweight='bold')
    ax_ent.invert_yaxis()
    ax_ent.legend(loc='lower right', fontsize=9)

    # Add delta annotations
    for i in range(n_heads):
        delta = entropy_delta[i]
        x_pos = max(entropy_s[i], entropy_st[i]) + 0.1
        ax_ent.text(x_pos, i, f'{delta:+.2f}', va='center', fontsize=8,
                   color=COLOR_EARLY if delta < -1 else '#555555',
                   fontweight='bold' if abs(delta) > 1 else 'normal')

    # ── Panel C: Attention sharpness (mean max attention) ──
    ax_attn = axes[2]
    ax_attn.barh(y_pos - bar_h/2, max_attn_s, bar_h, color=COLOR_S,
                 label='Max Attn (S)', edgecolor='white', linewidth=0.5, zorder=3)
    ax_attn.barh(y_pos + bar_h/2, max_attn_st, bar_h, color=COLOR_SST,
                 label='Max Attn (S+St)', edgecolor='white', linewidth=0.5, zorder=3)

    ax_attn.set_yticks(y_pos)
    ax_attn.set_yticklabels([])
    ax_attn.set_xlabel('Mean Max Attention Weight')
    ax_attn.set_title('C. Attention Sharpness', fontweight='bold')
    ax_attn.invert_yaxis()
    ax_attn.legend(loc='lower right', fontsize=9)

    # Highlight heads that become very sharp with structure
    for i in range(n_heads):
        if max_attn_st[i] > 0.8:
            x_pos = max_attn_st[i] + 0.01
            ax_attn.text(x_pos, i, f'{max_attn_st[i]:.2f}', va='center',
                        fontsize=8, color=COLOR_SST, fontweight='bold')

    fig.suptitle('Profile of Top-10 Structure-Responsive Heads',
                fontsize=TITLE_SIZE, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig7_top_heads')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('=' * 70)
    print('Phase 1 Figure Generation')
    print(f'Output directory: {OUTDIR}')
    print('=' * 70)

    figure1_cka_modality()
    figure2_sae_features()
    figure3_attention_atlas()
    figure4_jsd_by_layer()
    figure5_entropy_jsd()
    figure6_dssp_sae()
    figure7_top_heads()

    print('\n' + '=' * 70)
    print('All figures generated successfully!')
    print(f'Output directory: {OUTDIR}')

    # List all generated files
    generated = sorted(OUTDIR.glob('fig*'))
    print(f'\nGenerated {len(generated)} files:')
    for f in generated:
        size_kb = f.stat().st_size / 1024
        print(f'  {f.name:40s} ({size_kb:.1f} KB)')
    print('=' * 70)
