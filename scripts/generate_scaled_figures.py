#!/usr/bin/env python3
"""Generate publication-quality figures from scaled dataset results.

Run as: ./env/bin/python scripts/generate_scaled_figures.py

Generates figures from:
  - Phase 1 ESM-3 scaled (4,793 proteins): CKA, attention, functional attention,
    SS probing, SAE features, co-activation, decoder geometry
  - Phase 0 ESM-2 scaled (4,793 proteins): probing, SAE features, cross-model comparison

Output: results/figures/scaled/fig{N}_*.{png,pdf}
"""

import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# Style
# ============================================================
LABEL_SIZE = 12
TITLE_SIZE = 14
TICK_SIZE = 10
LEGEND_SIZE = 10

COLOR_S = '#2171b5'
COLOR_SST = '#e6550d'
COLOR_SF = '#31a354'
COLOR_SALL = '#756bb1'

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

OUTDIR = ROOT / "results" / "figures" / "scaled"
OUTDIR.mkdir(parents=True, exist_ok=True)
P1 = ROOT / "results" / "phase1_scaled"
P0 = ROOT / "results" / "phase0_scaled"


def save_fig(fig, name):
    fig.savefig(OUTDIR / f"{name}.png", dpi=300, facecolor="white", edgecolor="none")
    fig.savefig(OUTDIR / f"{name}.pdf", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Fig 1: CKA Modality Integration
# ============================================================
def fig1_cka_modality():
    with open(P1 / "modality_dropout" / "cka_results.json") as f:
        data = json.load(f)

    layers = data["layers"]
    conds = data["conditions"]  # S, S+St, S+F, S+St+F

    # Extract CKA(S, X) for each condition
    cka_s_sst = []
    cka_s_sf = []
    cka_s_sall = []
    cka_sst_sall = []

    for l in layers:
        m = data["cka_matrices"][str(l)]
        cka_s_sst.append(m[0][1])   # S vs S+St
        cka_s_sf.append(m[0][2])    # S vs S+F
        cka_s_sall.append(m[0][3])  # S vs S+St+F
        cka_sst_sall.append(m[1][3])  # S+St vs S+St+F

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, cka_s_sst, 'o-', color=COLOR_SST, lw=2, ms=6, label='CKA(S, S+St)')
    ax.plot(layers, cka_s_sf, 's-', color=COLOR_SF, lw=2, ms=6, label='CKA(S, S+F)')
    ax.plot(layers, cka_s_sall, 'D-', color=COLOR_SALL, lw=2, ms=6, label='CKA(S, S+St+F)')
    ax.plot(layers, cka_sst_sall, '^--', color='#d62728', lw=1.5, ms=5, alpha=0.7,
            label='CKA(S+St, S+St+F)')

    ax.axvspan(28, 40, alpha=0.1, color='orange', label='Integration zone')
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA Similarity')
    ax.set_title(f'Modality Integration in ESM-3 (n={data["num_proteins"]} proteins)')
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "fig1_cka_modality")


# ============================================================
# Fig 2: Attention Atlas (48×24 JSD heatmap)
# ============================================================
def fig2_attention_atlas():
    with open(P1 / "attention_atlas" / "attention_atlas.json") as f:
        data = json.load(f)

    jsd = np.array(data["jsd_matrix"])  # (48, 24)

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(jsd.T, aspect='auto', cmap='magma', origin='lower',
                   extent=[0, 47, 0, 23])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Head')
    ax.set_title(f'Attention Head Structure Responsiveness — JSD(S, S+St)\n'
                 f'(n={data["num_proteins"]} proteins)')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Jensen-Shannon Divergence')

    ax.set_xticks(range(0, 48, 4))
    ax.set_yticks(range(0, 24, 4))
    save_fig(fig, "fig2_attention_atlas")


# ============================================================
# Fig 3: JSD by Layer (mean across heads)
# ============================================================
def fig3_jsd_by_layer():
    with open(P1 / "attention_atlas" / "attention_atlas.json") as f:
        data = json.load(f)

    jsd = np.array(data["jsd_matrix"])  # (48, 24)
    mean_jsd = jsd.mean(axis=1)
    max_jsd = jsd.max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(range(48), 0, max_jsd, alpha=0.15, color=COLOR_SST, label='Max head JSD')
    ax.plot(range(48), mean_jsd, 'o-', color=COLOR_SST, lw=2, ms=4, label='Mean JSD')

    ax.axvspan(28, 40, alpha=0.1, color='orange', label='Integration zone')
    ax.axvspan(2, 9, alpha=0.1, color='red', label='Structure-responsive zone')
    ax.set_xlabel('Layer')
    ax.set_ylabel('JSD(S, S+St)')
    ax.set_title('Structure Responsiveness Peaks in Early Layers')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    save_fig(fig, "fig3_jsd_by_layer")


# ============================================================
# Fig 4: Functional Attention Enrichment
# ============================================================
def fig4_functional_attention():
    with open(P1 / "functional_attention" / "functional_attention_results.json") as f:
        data = json.load(f)

    layers = data["layers"]
    n_heads = data["n_heads"]

    # Build enrichment matrix (layers × heads)
    enrich_s = np.zeros((len(layers), n_heads))
    enrich_sst = np.zeros((len(layers), n_heads))
    for i, l in enumerate(layers):
        enrich_s[i] = data["enrichment_s"][str(l)]
        enrich_sst[i] = data["enrichment_sst"][str(l)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    vmin, vmax = 0.5, max(enrich_s.max(), enrich_sst.max())
    for ax, mat, title in [(axes[0], enrich_s, 'S-only'),
                            (axes[1], enrich_sst, 'S+St')]:
        im = ax.imshow(mat.T, aspect='auto', cmap='YlOrRd', origin='lower',
                       vmin=vmin, vmax=vmax)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Head')
        ax.set_title(f'Functional Enrichment — {title}')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_yticks(range(0, n_heads, 4))

    plt.colorbar(im, ax=axes, shrink=0.8, label='Enrichment (func / non-func)')
    fig.suptitle(f'Functional Site Attention Enrichment (n={data["num_proteins"]})',
                 fontsize=TITLE_SIZE, y=1.02)
    save_fig(fig, "fig4_functional_attention")


# ============================================================
# Fig 5: Functional Attention Summary by Layer
# ============================================================
def fig5_functional_summary():
    with open(P1 / "functional_attention" / "functional_attention_results.json") as f:
        data = json.load(f)

    layers = data["layers"]
    max_s = [max(data["enrichment_s"][str(l)]) for l in layers]
    max_sst = [max(data["enrichment_sst"][str(l)]) for l in layers]
    mean_s = [np.mean(data["enrichment_s"][str(l)]) for l in layers]
    mean_sst = [np.mean(data["enrichment_sst"][str(l)]) for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, max_s, w, color=COLOR_S, alpha=0.8, label='S-only (max head)')
    ax.bar(x + w/2, max_sst, w, color=COLOR_SST, alpha=0.8, label='S+St (max head)')
    ax.plot(x - w/2, mean_s, 'o-', color=COLOR_S, lw=1.5, ms=4)
    ax.plot(x + w/2, mean_sst, 'o-', color=COLOR_SST, lw=1.5, ms=4)

    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label='No enrichment')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Functional Enrichment')
    ax.set_title('Functional Attention: Two-Stage Detection')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    save_fig(fig, "fig5_functional_summary")


# ============================================================
# Fig 6: SS Probing — S vs S+St at 3 layers
# ============================================================
def fig6_ss_probing_comparison():
    with open(P1 / "ss_probing" / "ss_probing_results.json") as f:
        data = json.load(f)

    s_results = {d['layer']: d for d in data if d['condition'] == 'S'}
    sst_results = {d['layer']: d for d in data if d['condition'] == 'S+St'}

    layers = [16, 33, 42]
    s_acc = [s_results[l]['accuracy'] for l in layers]
    sst_acc = [sst_results[l]['accuracy'] for l in layers]
    s_f1 = [s_results[l]['f1_macro'] for l in layers]
    sst_f1 = [sst_results[l]['f1_macro'] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(layers))
    w = 0.35

    for ax, s_vals, sst_vals, ylabel, title in [
        (axes[0], s_acc, sst_acc, 'Accuracy', 'SS3 Classification Accuracy'),
        (axes[1], s_f1, sst_f1, 'Macro F1', 'SS3 Classification F1'),
    ]:
        bars1 = ax.bar(x - w/2, s_vals, w, color=COLOR_S, label='S-only')
        bars2 = ax.bar(x + w/2, sst_vals, w, color=COLOR_SST, label='S+St')
        ax.set_xlabel('Layer')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend()
        ax.set_ylim(0.7, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in list(zip(bars1, s_vals)) + list(zip(bars2, sst_vals)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    n_res = s_results[16]['n_train'] + s_results[16]['n_test']
    fig.suptitle(f'Secondary Structure Probing: S-only vs S+St (n={n_res:,} residues)',
                 fontsize=TITLE_SIZE, y=1.02)
    save_fig(fig, "fig6_ss_probing_comparison")


# ============================================================
# Fig 7: SS Probing — 9-layer S-only trajectory
# ============================================================
def fig7_ss_probing_9layers():
    with open(P1 / "ss_probing" / "ss_probing_results.json") as f:
        data = json.load(f)

    nine = sorted([d for d in data if d['condition'] == 'S_9layers'],
                  key=lambda d: d['layer'])
    layers = [d['layer'] for d in nine]
    acc = [d['accuracy'] for d in nine]
    f1 = [d['f1_macro'] for d in nine]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, acc, 'o-', color=COLOR_S, lw=2, ms=7, label='Accuracy')
    ax.plot(layers, f1, 's--', color='#6baed6', lw=2, ms=6, label='Macro F1')

    ax.axvspan(28, 40, alpha=0.1, color='orange', label='Integration zone')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Score')
    ax.set_title('ESM-3 SS3 Probing Across All Layers (S-only)')
    ax.set_xticks(layers)
    ax.set_ylim(0.55, 0.95)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate start and end
    ax.annotate(f'{acc[0]:.3f}', (layers[0], acc[0]), textcoords='offset points',
                xytext=(10, -15), fontsize=9)
    ax.annotate(f'{acc[-1]:.3f}', (layers[-1], acc[-1]), textcoords='offset points',
                xytext=(-30, 10), fontsize=9)
    save_fig(fig, "fig7_ss_probing_9layers")


# ============================================================
# Fig 8: SAE Feature Type Comparison (S vs S+St)
# ============================================================
def fig8_sae_features():
    with open(P1 / "modality_saes" / "all_annotations.json") as f:
        ann = json.load(f)

    def count_types(features):
        aa_specific = 0
        functional = 0
        structural = 0
        for feat in features:
            top_aa = feat.get('top_enriched_aa', [])
            func_e = feat.get('functional_site_enrichment', 1.0)
            ss_e = feat.get('ss_enrichment', {})
            if len(top_aa) > 0 and top_aa[0][1] > 3.0:
                aa_specific += 1
            if func_e > 2.0:
                functional += 1
            max_ss = max(ss_e.values()) if ss_e else 0
            if max_ss > 2.0:
                structural += 1
        return aa_specific, functional, structural

    configs = [
        ('S_layer_16', 'S L16'), ('S_layer_33', 'S L33'), ('S_layer_42', 'S L42'),
        ('S_St_layer_16', 'S+St L16'), ('S_St_layer_33', 'S+St L33'), ('S_St_layer_42', 'S+St L42'),
    ]

    labels = []
    aa_counts = []
    func_counts = []
    ss_counts = []
    for key, label in configs:
        if key in ann:
            aa, func, ss = count_types(ann[key])
            labels.append(label)
            aa_counts.append(aa)
            func_counts.append(func)
            ss_counts.append(ss)

    if not labels:
        print("  Skipping fig8: no SAE annotations found")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, aa_counts, w, color='#1f77b4', label='AA-specific (>3×)')
    ax.bar(x, func_counts, w, color='#ff7f0e', label='Functional (>2×)')
    ax.bar(x + w, ss_counts, w, color='#2ca02c', label='SS-enriched (>2×)')

    ax.set_xlabel('SAE Configuration')
    ax.set_ylabel('Feature Count (of 200)')
    ax.set_title('SAE Feature Types: S-only vs S+St Across Layers')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    save_fig(fig, "fig8_sae_features")


# ============================================================
# Fig 9: ESM-2 Probing Across Layers
# ============================================================
def fig9_esm2_probing():
    with open(P0 / "probing" / "probing_results.json") as f:
        data = json.load(f)

    # Group by property
    props = {}
    for d in data:
        p = d['property']
        if p not in props:
            props[p] = {'layers': [], 'values': []}
        props[p]['layers'].append(d['layer'])
        if d['task'] == 'regression':
            props[p]['values'].append(1 - d['mse'])  # 1-MSE for comparability
            props[p]['metric'] = '1 - MSE'
        else:
            props[p]['values'].append(d['accuracy'])
            props[p]['metric'] = 'Accuracy'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Classification tasks
    ax = axes[0]
    colors = {'amino_acid_identity': '#1f77b4', 'functional_site': '#ff7f0e',
              'secondary_structure_3': '#2ca02c'}
    labels_map = {'amino_acid_identity': 'AA Identity', 'functional_site': 'Functional Site',
                  'secondary_structure_3': 'SS3'}
    for p in ['amino_acid_identity', 'secondary_structure_3', 'functional_site']:
        if p in props:
            ax.plot(props[p]['layers'], props[p]['values'], 'o-', color=colors.get(p, 'gray'),
                    lw=2, ms=5, label=labels_map.get(p, p))
    ax.set_xlabel('ESM-2 Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title('ESM-2 Classification Probing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel B: Functional site AUROC
    ax = axes[1]
    func_data = [d for d in data if d['property'] == 'functional_site']
    func_data.sort(key=lambda x: x['layer'])
    layers_f = [d['layer'] for d in func_data]
    auroc = [d.get('auroc', 0) or 0 for d in func_data]
    ax.plot(layers_f, auroc, 'o-', color='#ff7f0e', lw=2, ms=6)
    ax.set_xlabel('ESM-2 Layer')
    ax.set_ylabel('AUROC')
    ax.set_title('Functional Site Detection (AUROC)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    ax.annotate(f'{auroc[0]:.3f}', (layers_f[0], auroc[0]),
                textcoords='offset points', xytext=(10, -10), fontsize=9)
    ax.annotate(f'{auroc[-1]:.3f}', (layers_f[-1], auroc[-1]),
                textcoords='offset points', xytext=(-30, 10), fontsize=9)

    fig.suptitle('ESM-2 (650M) Probing — Scaled Dataset (4,793 proteins)',
                 fontsize=TITLE_SIZE, y=1.02)
    save_fig(fig, "fig9_esm2_probing")


# ============================================================
# Fig 10: ESM-2 vs ESM-3 Cross-Model Feature Comparison
# ============================================================
def fig10_cross_model():
    with open(P0 / "cross_model_comparison.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Similarity distribution
    ax = axes[0]
    sim_stats = data["similarity_stats"]
    conv_counts = data["convergent_counts"]

    thresholds = sorted(conv_counts.keys(), key=float)
    fractions = [conv_counts[t] / data["n_esm2_features"] * 100 for t in thresholds]

    ax.bar(range(len(thresholds)), fractions, color='#4292c6', edgecolor='#2171b5', lw=1.2)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'>{t}' for t in thresholds])
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Convergent Features (%)')
    ax.set_title('Feature Convergence: ESM-2 ↔ ESM-3')
    ax.set_ylim(0, 105)
    for i, frac in enumerate(fractions):
        ax.text(i, frac + 1.5, f'{frac:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Feature type comparison
    ax = axes[1]
    ft_esm2 = data["feature_types"]["esm2"]
    ft_esm3 = data["feature_types"]["esm3"]
    types = sorted(set(list(ft_esm2.keys()) + list(ft_esm3.keys())))
    x = np.arange(len(types))
    w = 0.35
    ax.bar(x - w/2, [ft_esm2.get(t, 0) for t in types], w, color='#4292c6', label='ESM-2 L24')
    ax.bar(x + w/2, [ft_esm3.get(t, 0) for t in types], w, color=COLOR_SST, label='ESM-3 S L33')
    ax.set_xlabel('Feature Type')
    ax.set_ylabel('Count')
    ax.set_title('Feature Type Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=20, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    perm = data["permutation_test"]
    fig.suptitle(f'Cross-Model Feature Universality — {data["convergent_fraction_095"]:.1f}% convergent '
                 f'(z={perm["z_score"]:.1f}, p<0.0001)',
                 fontsize=TITLE_SIZE, y=1.02)
    save_fig(fig, "fig10_cross_model")


# ============================================================
# Fig 11: Co-activation Analysis
# ============================================================
def fig11_coactivation():
    coact_path = P1 / "coactivation" / "coactivation_results.json"
    if not coact_path.exists():
        print("  Skipping fig11: no coactivation data")
        return

    with open(coact_path) as f:
        data = json.load(f)

    configs = list(data.keys())
    if not configs:
        print("  Skipping fig11: empty coactivation data")
        return

    fig, axes = plt.subplots(1, min(len(configs), 3), figsize=(5 * min(len(configs), 3), 5))
    if not hasattr(axes, '__len__'):
        axes = [axes]

    for ax, cfg in zip(axes, configs[:3]):
        info = data[cfg]
        pairs = info.get("top_pairs", [])
        if not pairs:
            ax.set_title(f'{cfg}\n(no pairs)')
            continue
        pmis = [p["pmi"] for p in pairs[:30]]
        ax.barh(range(len(pmis)), pmis, color='#4292c6', edgecolor='#2171b5')
        ax.set_xlabel('PMI')
        ax.set_ylabel('Feature Pair Rank')
        ax.set_title(cfg.replace('_', ' '))
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle('Top Co-activating Feature Pairs', fontsize=TITLE_SIZE, y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig11_coactivation")


# ============================================================
# Fig 12: Decoder Geometry
# ============================================================
def fig12_decoder_geometry():
    geom_path = P1 / "decoder_geometry" / "decoder_geometry.json"
    if not geom_path.exists():
        print("  Skipping fig12: no decoder geometry data")
        return

    with open(geom_path) as f:
        data = json.load(f)

    configs = list(data.keys())
    if not configs:
        print("  Skipping fig12: empty data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Mean cosine similarity
    ax = axes[0]
    sst_cfgs = [c for c in configs if c.startswith('S_St_')]
    s_cfgs = [c for c in configs if c.startswith('S_') and c not in sst_cfgs]
    s_layers = [c.split('_')[-1] for c in s_cfgs]
    s_cos = [data[c]["mean_cosine_sim"] for c in s_cfgs]
    sst_cos = [data[c]["mean_cosine_sim"] for c in sst_cfgs]

    x = np.arange(len(s_layers))
    w = 0.35
    ax.bar(x - w/2, s_cos, w, color=COLOR_S, label='S-only')
    ax.bar(x + w/2, sst_cos, w, color=COLOR_SST, label='S+St')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Cosine Similarity')
    ax.set_title('Decoder Weight Similarity')
    ax.set_xticks(x)
    ax.set_xticklabels(s_layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Effective dimensionality
    ax = axes[1]
    s_dim = [data[c]["effective_dimensionality"] for c in s_cfgs]
    sst_dim = [data[c]["effective_dimensionality"] for c in sst_cfgs]
    ax.bar(x - w/2, s_dim, w, color=COLOR_S, label='S-only')
    ax.bar(x + w/2, sst_dim, w, color=COLOR_SST, label='S+St')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Effective Dimensionality')
    ax.set_title('Feature Space Dimensionality')
    ax.set_xticks(x)
    ax.set_xticklabels(s_layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('SAE Decoder Geometry Analysis', fontsize=TITLE_SIZE, y=1.02)
    save_fig(fig, "fig12_decoder_geometry")


# ============================================================
# Fig 13: ESM-2 vs ESM-3 SS Probing Comparison
# ============================================================
def fig13_esm2_vs_esm3_ss():
    # ESM-3 9-layer probing
    with open(P1 / "ss_probing" / "ss_probing_results.json") as f:
        esm3_data = json.load(f)

    esm3_9 = sorted([d for d in esm3_data if d['condition'] == 'S_9layers'],
                     key=lambda d: d['layer'])
    esm3_layers = [d['layer'] for d in esm3_9]
    esm3_acc = [d['accuracy'] for d in esm3_9]

    # ESM-2 probing
    with open(P0 / "probing" / "probing_results.json") as f:
        esm2_data = json.load(f)

    esm2_ss = sorted([d for d in esm2_data if d['property'] == 'secondary_structure_3'],
                      key=lambda d: d['layer'])
    esm2_layers = [d['layer'] for d in esm2_ss]
    esm2_acc = [d['accuracy'] for d in esm2_ss]

    # Normalize layers to fraction of total depth
    esm3_frac = [l / 47 for l in esm3_layers]
    esm2_frac = [l / 32 for l in esm2_layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(esm2_frac, esm2_acc, 'o-', color='#4292c6', lw=2, ms=6,
            label=f'ESM-2 (650M, 33 layers)')
    ax.plot(esm3_frac, esm3_acc, 's-', color=COLOR_SST, lw=2, ms=6,
            label=f'ESM-3 S-only (1.4B, 48 layers)')
    ax.set_xlabel('Relative Depth (layer / total)')
    ax.set_ylabel('SS3 Accuracy')
    ax.set_title('Secondary Structure Probing: ESM-2 vs ESM-3')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.55, 0.95)
    save_fig(fig, "fig13_esm2_vs_esm3_ss")


# ============================================================
# Fig 14: Combined Summary — Key Findings
# ============================================================
def fig14_summary():
    # Load all data for summary
    with open(P1 / "modality_dropout" / "cka_results.json") as f:
        cka = json.load(f)
    with open(P1 / "attention_atlas" / "attention_atlas.json") as f:
        atlas = json.load(f)
    with open(P1 / "ss_probing" / "ss_probing_results.json") as f:
        probing = json.load(f)
    with open(P0 / "cross_model_comparison.json") as f:
        xmodel = json.load(f)

    jsd = np.array(atlas["jsd_matrix"])
    mean_jsd = jsd.mean(axis=1)

    # CKA(S, S+St) across layers
    cka_layers = cka["layers"]
    cka_s_sst = [cka["cka_matrices"][str(l)][0][1] for l in cka_layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A: CKA + JSD overlay
    ax = axes[0, 0]
    ax.plot(cka_layers, cka_s_sst, 'o-', color=COLOR_SST, lw=2, ms=6, label='CKA(S, S+St)')
    ax2 = ax.twinx()
    ax2.plot(range(48), mean_jsd, '-', color='#d62728', lw=1.5, alpha=0.7, label='Mean JSD')
    ax.set_xlabel('Layer')
    ax.set_ylabel('CKA', color=COLOR_SST)
    ax2.set_ylabel('JSD', color='#d62728')
    ax.set_title('A. Representation vs Attention Integration')
    ax.axvspan(28, 40, alpha=0.1, color='orange')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')

    # Panel B: SS probing S vs S+St
    ax = axes[0, 1]
    s_data = {d['layer']: d for d in probing if d['condition'] == 'S'}
    sst_data = {d['layer']: d for d in probing if d['condition'] == 'S+St'}
    layers_3 = [16, 33, 42]
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, [s_data[l]['accuracy'] for l in layers_3], w, color=COLOR_S, label='S-only')
    ax.bar(x + w/2, [sst_data[l]['accuracy'] for l in layers_3], w, color=COLOR_SST, label='S+St')
    ax.set_xlabel('Layer')
    ax.set_ylabel('SS3 Accuracy')
    ax.set_title('B. Structure Tokens Boost SS Readability')
    ax.set_xticks(x)
    ax.set_xticklabels(layers_3)
    ax.legend(fontsize=9)
    ax.set_ylim(0.7, 1.0)

    # Panel C: 9-layer trajectory
    ax = axes[1, 0]
    nine = sorted([d for d in probing if d['condition'] == 'S_9layers'], key=lambda d: d['layer'])
    ax.plot([d['layer'] for d in nine], [d['accuracy'] for d in nine],
            'o-', color=COLOR_S, lw=2, ms=6)
    ax.axvspan(28, 40, alpha=0.1, color='orange', label='Integration zone')
    ax.set_xlabel('Layer')
    ax.set_ylabel('SS3 Accuracy')
    ax.set_title('C. Structural Knowledge Emerges Progressively')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel D: Cross-model convergence
    ax = axes[1, 1]
    conv = xmodel["convergent_counts"]
    thresholds = sorted(conv.keys(), key=float)
    fracs = [conv[t] / xmodel["n_esm2_features"] * 100 for t in thresholds]
    ax.bar(range(len(thresholds)), fracs, color='#4292c6', edgecolor='#2171b5')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'>{t}' for t in thresholds])
    ax.set_xlabel('Similarity Threshold')
    ax.set_ylabel('Convergent (%)')
    ax.set_title(f'D. ESM-2 ↔ ESM-3 Feature Universality ({xmodel["convergent_fraction_095"]:.1f}%)')
    ax.set_ylim(0, 105)

    fig.suptitle('Mechanistic Interpretability of ESM-3: Key Findings (n=4,793 proteins)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig14_summary")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating scaled dataset figures...")
    print(f"Output: {OUTDIR}")
    print()

    fig1_cka_modality()
    fig2_attention_atlas()
    fig3_jsd_by_layer()
    fig4_functional_attention()
    fig5_functional_summary()
    fig6_ss_probing_comparison()
    fig7_ss_probing_9layers()
    fig8_sae_features()
    fig9_esm2_probing()
    fig10_cross_model()
    fig11_coactivation()
    fig12_decoder_geometry()
    fig13_esm2_vs_esm3_ss()
    fig14_summary()

    print(f"\nDone! {len(list(OUTDIR.glob('*.png')))} figures generated.")
