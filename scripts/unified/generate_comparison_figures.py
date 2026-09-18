#!/usr/bin/env python3
"""Generate publication-quality comparison figures for unified analysis.

Creates side-by-side panels showing ESM-2 vs ESM-3 for all completed analyses.

Run as:
    ./env/bin/python scripts/unified/generate_comparison_figures.py --models esm2 esm3

Output: results/figures/unified/fig{N}_{name}.{png,pdf}
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


FIG_DIR = ROOT / "results" / "figures" / "unified"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "esm2": "#2196F3",
    "esm3": "#FF5722",
}

MODEL_LABELS = {
    "esm2": "ESM-2 (650M)",
    "esm3": "ESM-3 (1.4B)",
}


def load_model_results(model_name):
    """Load all available results for a model."""
    model_dir = ROOT / "results" / "unified" / model_name
    results = {}
    if not model_dir.exists():
        return results

    for fname in model_dir.glob("*.json"):
        with open(fname) as f:
            results[fname.stem] = json.load(f)
    return results


# ============================================================
# Figure 1: Layer Ablation Profiles
# ============================================================

def fig_layer_ablation(results_dict, models, fig_num=1):
    """Layer ablation profiles side by side."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for model in models:
        if model not in results_dict or "layer_ablation" not in results_dict[model]:
            continue
        data = results_dict[model]["layer_ablation"]
        per_layer = data["per_layer"]
        n_layers = data["n_layers"]

        layers = sorted(int(k) for k in per_layer.keys())
        kl_vals = [per_layer[str(l)]["mean_kl"] for l in layers]
        layer_fracs = [l / (n_layers - 1) for l in layers]

        ax.plot(layer_fracs, kl_vals, "o-",
                color=MODEL_COLORS.get(model, "gray"),
                label=MODEL_LABELS.get(model, model),
                markersize=3, linewidth=1.5)

    ax.set_xlabel("Relative Layer Position", fontsize=12)
    ax.set_ylabel("Mean KL Divergence (ablated vs original)", fontsize=12)
    ax.set_title("Layer Ablation Impact", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"fig{fig_num}_layer_ablation")
    plt.close()


# ============================================================
# Figure 2: Head Ablation Heatmaps
# ============================================================

def fig_head_ablation(results_dict, models, fig_num=2):
    """Head ablation importance heatmaps."""
    n_models = sum(1 for m in models if m in results_dict and "head_ablation" in results_dict[m])
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax_idx, model in enumerate(m for m in models if m in results_dict and "head_ablation" in results_dict[m]):
        data = results_dict[model]["head_ablation"]
        per_head = data["per_head"]
        ablation_layers = data["ablation_layers"]
        n_heads = data["n_heads"]

        # Build heatmap
        heatmap = np.zeros((len(ablation_layers), n_heads))
        for i, layer_idx in enumerate(ablation_layers):
            for h in range(n_heads):
                key = f"L{layer_idx}_H{h}"
                if key in per_head:
                    heatmap[i, h] = per_head[key]["mean_kl"]

        im = axes[ax_idx].imshow(heatmap, aspect="auto", cmap="Reds")
        axes[ax_idx].set_xlabel("Head", fontsize=11)
        axes[ax_idx].set_ylabel("Layer", fontsize=11)
        axes[ax_idx].set_yticks(range(len(ablation_layers)))
        axes[ax_idx].set_yticklabels(ablation_layers)
        axes[ax_idx].set_title(MODEL_LABELS.get(model, model), fontsize=13)
        plt.colorbar(im, ax=axes[ax_idx], label="KL Divergence")

    fig.suptitle("Head Ablation Impact", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_head_ablation")
    plt.close()


# ============================================================
# Figure 3: Attention Properties
# ============================================================

def fig_attention_properties(results_dict, models, fig_num=3):
    """Attention entropy and local fraction by layer."""
    n_models = sum(1 for m in models if m in results_dict and "attention_analysis" in results_dict[m])
    if n_models == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model in models:
        if model not in results_dict or "attention_analysis" not in results_dict[model]:
            continue
        data = results_dict[model]["attention_analysis"]
        per_head = data["per_head"]
        n_layers = data["n_layers"]
        n_heads = data["n_heads"]

        # Per-layer averages
        layer_entropy = []
        layer_local = []
        for l in range(n_layers):
            ents = [per_head[f"L{l}_H{h}"]["entropy"] for h in range(n_heads)]
            locs = [per_head[f"L{l}_H{h}"]["local_fraction"] for h in range(n_heads)]
            layer_entropy.append(np.mean(ents))
            layer_local.append(np.mean(locs))

        x = np.arange(n_layers) / (n_layers - 1)
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        ax1.plot(x, layer_entropy, "-", color=color, label=label, linewidth=1.5)
        ax2.plot(x, layer_local, "-", color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Relative Layer Position")
    ax1.set_ylabel("Mean Attention Entropy (nats)")
    ax1.set_title("Attention Entropy by Layer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Layer Position")
    ax2.set_ylabel("Local Attention Fraction (+-5)")
    ax2.set_title("Local vs Global Attention")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_attention_properties")
    plt.close()


# ============================================================
# Figure 4: Activation Patching Curves
# ============================================================

def fig_activation_patching(results_dict, models, fig_num=4):
    """Activation patching effect curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for model in models:
        if model not in results_dict or "activation_patching" not in results_dict[model]:
            continue
        data = results_dict[model]["activation_patching"]
        per_layer = data["per_layer"]

        # Estimate total layers from max layer index
        max_layer = max(int(k) for k in per_layer.keys())
        layers = sorted(int(k) for k in per_layer.keys())
        kl_vals = [per_layer[str(l)]["mean_kl"] for l in layers]
        layer_fracs = [l / max_layer for l in layers]

        ax.plot(layer_fracs, kl_vals, "o-",
                color=MODEL_COLORS.get(model, "gray"),
                label=MODEL_LABELS.get(model, model),
                markersize=4, linewidth=1.5)

    ax.set_xlabel("Relative Layer Position", fontsize=12)
    ax.set_ylabel("Mean KL Divergence (patched vs original)", fontsize=12)
    ax.set_title("Cross-Property Activation Patching", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"fig{fig_num}_activation_patching")
    plt.close()


# ============================================================
# Figure 5: CKA Within-Model Heatmaps
# ============================================================

def fig_cka_within_model(results_dict, models, fig_num=5):
    """Within-model CKA heatmaps."""
    n_models = sum(1 for m in models if m in results_dict and "cka_within_model" in results_dict[m])
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax_idx, model in enumerate(m for m in models if m in results_dict and "cka_within_model" in results_dict[m]):
        data = results_dict[model]["cka_within_model"]
        cka_matrix = np.array(data["cka_matrix"])
        layers = data["layers"]

        im = axes[ax_idx].imshow(cka_matrix, vmin=0, vmax=1, cmap="viridis")
        axes[ax_idx].set_xlabel("Layer")
        axes[ax_idx].set_ylabel("Layer")
        n = len(layers)
        tick_step = max(1, n // 6)
        axes[ax_idx].set_xticks(range(0, n, tick_step))
        axes[ax_idx].set_xticklabels([str(layers[i]) for i in range(0, n, tick_step)])
        axes[ax_idx].set_yticks(range(0, n, tick_step))
        axes[ax_idx].set_yticklabels([str(layers[i]) for i in range(0, n, tick_step)])
        axes[ax_idx].set_title(MODEL_LABELS.get(model, model), fontsize=13)
        plt.colorbar(im, ax=axes[ax_idx], label="CKA")

    fig.suptitle("Within-Model Layer CKA", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_cka_within_model")
    plt.close()


# ============================================================
# Figure 6: SAE Feature Causal Importance
# ============================================================

def fig_sae_feature_ablation(results_dict, models, fig_num=6):
    """SAE feature causal importance distributions."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for model in models:
        if model not in results_dict:
            continue
        # Find SAE ablation results
        sae_keys = [k for k in results_dict[model] if k.startswith("sae_feature_ablation")]
        if not sae_keys:
            continue

        for sae_key in sae_keys:
            data = results_dict[model][sae_key]
            feat_results = data["feature_results"]

            kl_vals = sorted([v["mean_kl"] for v in feat_results.values()], reverse=True)
            color = MODEL_COLORS.get(model, "gray")
            label = f"{MODEL_LABELS.get(model, model)} (L{data['layer']})"
            ax.plot(range(len(kl_vals)), kl_vals, "-",
                    color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Feature Rank (by causal importance)", fontsize=12)
    ax.set_ylabel("KL Divergence when ablated", fontsize=12)
    ax.set_title("SAE Feature Causal Importance", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"fig{fig_num}_sae_feature_ablation")
    plt.close()


# ============================================================
# Figure 7: Logit Lens — Prediction Formation
# ============================================================

def fig_logit_lens(results_dict, models, fig_num=7):
    """Raw logit lens: KL divergence and top-1 agreement across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model in models:
        if model not in results_dict or "logit_lens" not in results_dict[model]:
            continue
        data = results_dict[model]["logit_lens"]
        per_layer = data["per_layer"]
        n_layers = data["n_layers"]

        layers = sorted(int(k) for k in per_layer.keys())
        kl_vals = [per_layer[str(l)]["mean_kl"] for l in layers]
        top1_vals = [per_layer[str(l)]["mean_top1_agreement"] for l in layers]
        x = [l / (n_layers - 1) for l in layers]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        ax1.plot(x, kl_vals, "-", color=color, label=label, linewidth=1.5)
        ax2.plot(x, top1_vals, "-", color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Relative Layer Position")
    ax1.set_ylabel("KL Divergence from Final Logits")
    ax1.set_title("Logit Lens: Prediction Distance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Layer Position")
    ax2.set_ylabel("Top-1 Agreement with Final")
    ax2.set_title("Logit Lens: Token Prediction Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_logit_lens")
    plt.close()


# ============================================================
# Figure 8: Tuned Lens — Raw vs Learned Projection
# ============================================================

def fig_tuned_lens(results_dict, models, fig_num=8):
    """Tuned lens: raw vs tuned KL divergence and top-1 agreement."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model in models:
        if model not in results_dict or "tuned_lens" not in results_dict[model]:
            continue
        data = results_dict[model]["tuned_lens"]
        per_layer = data["per_layer"]
        n_layers = data["n_layers"]

        layers = sorted(int(k) for k in per_layer.keys())
        def _val(d):
            return d["mean"] if isinstance(d, dict) else d
        raw_kl = [_val(per_layer[str(l)]["raw_kl"]) for l in layers]
        tuned_kl = [_val(per_layer[str(l)]["tuned_kl"]) for l in layers]
        raw_top1 = [_val(per_layer[str(l)]["raw_top1"]) for l in layers]
        tuned_top1 = [_val(per_layer[str(l)]["tuned_top1"]) for l in layers]
        x = [l / (n_layers - 1) for l in layers]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        ax1.plot(x, raw_kl, "--", color=color, alpha=0.5, linewidth=1)
        ax1.plot(x, tuned_kl, "-", color=color, label=label, linewidth=1.5)
        ax2.plot(x, raw_top1, "--", color=color, alpha=0.5, linewidth=1)
        ax2.plot(x, tuned_top1, "-", color=color, label=label, linewidth=1.5)

    # Add legend entries for line styles
    ax1.plot([], [], "--", color="gray", alpha=0.5, label="Raw (dashed)")
    ax1.plot([], [], "-", color="gray", label="Tuned (solid)")
    ax1.set_xlabel("Relative Layer Position")
    ax1.set_ylabel("KL Divergence from Final Logits")
    ax1.set_title("Tuned Lens: KL Reduction")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot([], [], "--", color="gray", alpha=0.5, label="Raw (dashed)")
    ax2.plot([], [], "-", color="gray", label="Tuned (solid)")
    ax2.set_xlabel("Relative Layer Position")
    ax2.set_ylabel("Top-1 Agreement with Final")
    ax2.set_title("Tuned Lens: Prediction Accuracy")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_tuned_lens")
    plt.close()


# ============================================================
# Figure 9: Direct Logit Attribution
# ============================================================

def fig_direct_logit_attribution(results_dict, models, fig_num=9):
    """Per-layer logit contribution: norm, cosine, and attn/MLP breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for model in models:
        if model not in results_dict or "direct_logit_attribution" not in results_dict[model]:
            continue
        data = results_dict[model]["direct_logit_attribution"]
        n_layers = data["n_layers"]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        # Panel 1: Per-layer logit norm
        norms = data["per_layer_logit_norm"]
        layers = sorted(int(k) for k in norms.keys())
        vals = [norms[str(l)]["mean"] for l in layers]
        x = [l / (n_layers - 1) for l in layers]
        axes[0].plot(x, vals, "-", color=color, label=label, linewidth=1.5)

        # Panel 2: Cosine similarity with final logits
        cosines = data["per_layer_cosine_with_final"]
        layers_c = sorted(int(k) for k in cosines.keys())
        vals_c = [cosines[str(l)]["mean"] for l in layers_c]
        x_c = [l / (n_layers - 1) for l in layers_c]
        axes[1].plot(x_c, vals_c, "-", color=color, label=label, linewidth=1.5)

        # Panel 3: Attn vs MLP breakdown (bar chart)
        attn_norms = data.get("attn_logit_norm", {})
        mlp_norms = data.get("mlp_logit_norm", {})
        if attn_norms and mlp_norms:
            comp_layers = sorted(int(k) for k in attn_norms.keys())
            attn_vals = [attn_norms[str(l)]["mean"] for l in comp_layers]
            mlp_vals = [mlp_norms[str(l)]["mean"] for l in comp_layers]
            x_pos = np.arange(len(comp_layers))
            width = 0.35
            offset = -width / 2 if model == "esm2" else width / 2
            axes[2].bar(x_pos + offset, attn_vals, width * 0.45,
                       color=color, alpha=0.7, label=f"{label} attn")
            axes[2].bar(x_pos + offset, mlp_vals, width * 0.45,
                       bottom=attn_vals, color=color, alpha=0.4)
            axes[2].set_xticks(x_pos)
            axes[2].set_xticklabels([str(l) for l in comp_layers], fontsize=9)

    axes[0].set_xlabel("Relative Layer Position")
    axes[0].set_ylabel("Logit Contribution Norm")
    axes[0].set_title("Per-Layer Logit Contribution")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Relative Layer Position")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Alignment with Final Logits")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Logit Norm (stacked: attn + MLP)")
    axes[2].set_title("Attn vs MLP Logit Contribution")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_direct_logit_attribution")
    plt.close()


# ============================================================
# Figure 10: Residual Stream Decomposition
# ============================================================

def fig_residual_decomposition(results_dict, models, fig_num=10):
    """Attention vs MLP contribution fraction across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model in models:
        if model not in results_dict or "residual_decomposition" not in results_dict[model]:
            continue
        data = results_dict[model]["residual_decomposition"]
        per_layer = data["per_layer"]
        n_layers = data["n_layers"]

        layers = sorted(int(k) for k in per_layer.keys())
        def _v(d):
            return d["mean"] if isinstance(d, dict) else d
        attn_frac = [_v(per_layer[str(l)]["attn_fraction"]) for l in layers]
        mlp_frac = [_v(per_layer[str(l)]["mlp_fraction"]) for l in layers]
        cosine = [_v(per_layer[str(l)]["attn_mlp_cosine"]) for l in layers]
        x = [l / (n_layers - 1) for l in layers]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        ax1.plot(x, attn_frac, "-", color=color, label=f"{label} (attn)", linewidth=1.5)
        ax1.plot(x, mlp_frac, "--", color=color, label=f"{label} (MLP)", linewidth=1.5, alpha=0.7)
        ax2.plot(x, cosine, "-", color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Relative Layer Position")
    ax1.set_ylabel("Fraction of Residual Norm")
    ax1.set_title("Attention vs MLP Contribution")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Layer Position")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Attn-MLP Output Alignment")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_residual_decomposition")
    plt.close()


# ============================================================
# Figure 11: OV/QK Circuit Decomposition
# ============================================================

def fig_ov_qk_decomposition(results_dict, models, fig_num=11):
    """OV/QK effective rank and head type distribution across layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model in models:
        if model not in results_dict or "ov_qk_decomposition" not in results_dict[model]:
            continue
        data = results_dict[model]["ov_qk_decomposition"]
        per_layer = data["per_layer"]
        n_layers = data["n_layers"]

        layers = sorted(int(k) for k in per_layer.keys())
        ov_ranks = [per_layer[str(l)]["mean_ov_rank"] for l in layers]
        qk_ranks = [per_layer[str(l)]["mean_qk_rank"] for l in layers]
        x = [l / (n_layers - 1) for l in layers]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        ax1.plot(x, ov_ranks, "-", color=color, label=f"{label} (OV)", linewidth=1.5)
        ax1.plot(x, qk_ranks, "--", color=color, label=f"{label} (QK)",
                linewidth=1.5, alpha=0.7)

        # Spectral concentration per layer (mean across heads)
        concentrations = []
        for l in layers:
            heads = per_layer[str(l)]["heads"]
            concs = [heads[str(h)]["spectral_concentration"] for h in range(len(heads))]
            concentrations.append(np.mean(concs))
        ax2.plot(x, concentrations, "-", color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Relative Layer Position")
    ax1.set_ylabel("Effective Rank")
    ax1.set_title("OV/QK Matrix Effective Rank")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Relative Layer Position")
    ax2.set_ylabel("Spectral Concentration (top-5)")
    ax2.set_title("OV Spectral Concentration")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_ov_qk_decomposition")
    plt.close()


# ============================================================
# Figure 12: Attribution Patching
# ============================================================

def fig_attribution_patching(results_dict, models, fig_num=12):
    """Gradient-based attribution patching effect per layer."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for model in models:
        if model not in results_dict or "attribution_patching" not in results_dict[model]:
            continue
        data = results_dict[model]["attribution_patching"]
        per_layer = data["per_layer"]
        n_layers = data["n_layers"]

        layers = sorted(int(k) for k in per_layer.keys())
        effects = [per_layer[str(l)]["normalized_effect"] for l in layers]
        x = [l / (n_layers - 1) for l in layers]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        ax.plot(x, effects, "o-", color=color, label=label, markersize=3, linewidth=1.5)
        # Mark peak
        peak_idx = np.argmax(effects)
        ax.plot(x[peak_idx], effects[peak_idx], "*", color=color, markersize=12)

    ax.set_xlabel("Relative Layer Position", fontsize=12)
    ax.set_ylabel("Normalized Attribution Effect", fontsize=12)
    ax.set_title("Attribution Patching: Layer-wise Effects", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_figure(fig, f"fig{fig_num}_attribution_patching")
    plt.close()


# ============================================================
# Figure 13: Mutation Sensitivity
# ============================================================

def fig_mutation_sensitivity(results_dict, models, fig_num=13):
    """Mutation sensitivity: functional vs non-functional sites + SS breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bar_data = []
    for model in models:
        if model not in results_dict or "mutation_sensitivity" not in results_dict[model]:
            continue
        data = results_dict[model]["mutation_sensitivity"]
        fvn = data["functional_vs_nonfunctional"]
        label = MODEL_LABELS.get(model, model)
        color = MODEL_COLORS.get(model, "gray")
        bar_data.append((model, label, color, fvn))

    if bar_data:
        x_pos = np.arange(len(bar_data))
        width = 0.35
        for i, (model, label, color, fvn) in enumerate(bar_data):
            func_kl = fvn.get("functional", {}).get("mean", 0)
            nonfunc_kl = fvn.get("nonfunctional", {}).get("mean", 0)
            ax1.bar(i - width / 2, func_kl, width, color=color, alpha=0.8, label="Functional" if i == 0 else "")
            ax1.bar(i + width / 2, nonfunc_kl, width, color=color, alpha=0.4, label="Non-functional" if i == 0 else "")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([d[1] for d in bar_data])
        ax1.set_ylabel("Mean KL Divergence (mutation effect)")
        ax1.set_title("Mutation Sensitivity by Site Type")
        ax1.legend()

    # SS breakdown
    for model in models:
        if model not in results_dict or "mutation_sensitivity" not in results_dict[model]:
            continue
        data = results_dict[model]["mutation_sensitivity"]
        ss = data.get("secondary_structure", {})
        if not ss:
            continue
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ss_types = sorted(ss.keys())
        vals = [ss[s]["mean"] for s in ss_types]
        x_pos = np.arange(len(ss_types))
        offset = -0.2 if model == "esm2" else 0.2
        ax2.bar(x_pos + offset, vals, 0.35, color=color, alpha=0.7, label=label)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([s.capitalize() for s in ss_types])

    ax2.set_ylabel("Mean KL Divergence")
    ax2.set_title("Mutation Sensitivity by Secondary Structure")
    ax2.legend()

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_mutation_sensitivity")
    plt.close()


# ============================================================
# Figure 14: Causal Probing
# ============================================================

def fig_causal_probing(results_dict, models, fig_num=14):
    """Causal probing: interchange intervention results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, model in enumerate(models):
        if model not in results_dict or "causal_probing" not in results_dict[model]:
            continue
        data = results_dict[model]["causal_probing"]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        # Panel 1: KL values for H→S, S→H, and random
        h2s_kl = data["helix_to_sheet_kl"]["mean"]
        s2h_kl = data["sheet_to_helix_kl"]["mean"]
        rand_kl = data["random_direction_kl"]["mean"]

        x_pos = np.arange(3)
        offset = -0.2 if model == "esm2" else 0.2
        ax1.bar(x_pos + offset, [h2s_kl, s2h_kl, rand_kl], 0.35,
               color=color, alpha=0.7, label=label)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(["Helix\u2192Sheet", "Sheet\u2192Helix", "Random"])

        # Panel 2: Causal ratio and probe accuracy
        ax2.bar(i * 2, data["causal_ratio"], 0.8, color=color, alpha=0.7, label=f"{label} ratio")
        ax2.bar(i * 2 + 1, data["probe_accuracy"] * 10, 0.8, color=color, alpha=0.4)
        # Text annotations
        ax2.text(i * 2, data["causal_ratio"] + 0.2, f'{data["causal_ratio"]:.1f}',
                ha="center", fontsize=9)
        ax2.text(i * 2 + 1, data["probe_accuracy"] * 10 + 0.2,
                f'{data["probe_accuracy"]:.1%}', ha="center", fontsize=9)

    ax1.set_ylabel("Mean KL Divergence")
    ax1.set_title("Interchange Intervention: Direction Effect")
    ax1.legend()

    ax2.set_ylabel("Value (ratio / accuracy x10)")
    ax2.set_title("Causal Ratio & Probe Accuracy")
    ax2.set_xticks(range(len(models) * 2))
    labels = []
    for m in models:
        labels.extend([f"{MODEL_LABELS.get(m, m)}\nRatio", f"{MODEL_LABELS.get(m, m)}\nAccuracy"])
    ax2.set_xticklabels(labels, fontsize=8)

    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_causal_probing")
    plt.close()


# ============================================================
# Figure 15: Contact Map Precision (if data available)
# ============================================================

def fig_contact_map(results_dict, models, fig_num=15):
    """Contact map precision from attention weights across layers."""
    has_data = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("L", "P@L"), ("L/2", "P@L/2"), ("L/5", "P@L/5")]

    for model in models:
        if model not in results_dict or "contact_map" not in results_dict[model]:
            continue
        data = results_dict[model]["contact_map"]
        per_layer = data.get("per_layer", {})
        if not per_layer:
            continue
        has_data = True
        n_layers = data["n_layers"]

        layers = sorted(int(k.replace("L", "")) if k.startswith("L") else int(k)
                        for k in per_layer.keys())
        x = [l / (n_layers - 1) for l in layers]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        for i, (key, title) in enumerate(metrics):
            vals = []
            for l in layers:
                lk = str(l)
                if lk not in per_layer:
                    lk = f"L{l}"
                v = per_layer.get(lk, {})
                vals.append(v.get(key, v.get(f"precision_{key}", 0)))
            axes[i].plot(x, vals, "o-", color=color, label=label,
                        markersize=4, linewidth=1.5)

    if not has_data:
        plt.close()
        return

    for i, (key, title) in enumerate(metrics):
        axes[i].set_xlabel("Relative Layer Position")
        axes[i].set_ylabel(title)
        axes[i].set_title(title)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("Contact Map Prediction from Attention", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_contact_map")
    plt.close()


# ============================================================
# Figure 16: Steering Vectors (if data available)
# ============================================================

def fig_steering_vectors(results_dict, models, fig_num=16):
    """Steering vector dose-response curves across scales."""
    has_data = False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vec_names = ["helix_vs_sheet", "helix_vs_coil", "sheet_vs_coil"]
    vec_titles = ["Helix vs Sheet", "Helix vs Coil", "Sheet vs Coil"]

    for model in models:
        if model not in results_dict or "steering_vectors" not in results_dict[model]:
            continue
        data = results_dict[model]["steering_vectors"]
        evaluation = data.get("evaluation", {})
        if not evaluation:
            continue
        has_data = True
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        for i, vec_name in enumerate(vec_names):
            if vec_name not in evaluation:
                continue
            per_scale = evaluation[vec_name].get("per_scale", {})
            scales = sorted(float(s) for s in per_scale.keys())
            kl_vals = [per_scale[str(s)]["mean_kl"] for s in scales]
            axes[i].plot(scales, kl_vals, "o-", color=color, label=label,
                        markersize=4, linewidth=1.5)

    if not has_data:
        plt.close()
        return

    for i, title in enumerate(vec_titles):
        axes[i].set_xlabel("Steering Scale")
        axes[i].set_ylabel("Mean KL Divergence")
        axes[i].set_title(title)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)

    fig.suptitle("Steering Vector Dose-Response", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_steering_vectors")
    plt.close()


# ============================================================
# Figure 17: Sparse Feature Circuits (if data available)
# ============================================================

def fig_sparse_feature_circuits(results_dict, models, fig_num=17):
    """Sparse feature circuit structure across layer pairs.

    Shows: (a) mean |delta| effect size per layer pair,
           (b) effect size distributions (box plots),
           (c) top-feature hub connectivity (max fan-out from upstream).
    """
    has_data = False

    model_stats = {}
    for model in models:
        if model not in results_dict or "sparse_feature_circuits" not in results_dict[model]:
            continue
        data = results_dict[model]["sparse_feature_circuits"]
        layer_pairs = data.get("layer_pairs", {})
        if not layer_pairs:
            continue
        has_data = True
        stats = []
        for pair_key, pair_data in layer_pairs.items():
            conns = pair_data.get("connections", [])
            all_deltas = []
            upstream_fan_out = {}  # upstream_id -> count of downstream targets
            for c in conns:
                for uid, udata in c.get("upstream_connections", {}).items():
                    all_deltas.append(abs(udata["mean_delta"]))
                    upstream_fan_out[uid] = upstream_fan_out.get(uid, 0) + 1
            top_hubs = sorted(upstream_fan_out.values(), reverse=True)[:10]
            stats.append({
                "pair_key": pair_key,
                "deltas": all_deltas,
                "mean_delta": float(np.mean(all_deltas)) if all_deltas else 0,
                "median_delta": float(np.median(all_deltas)) if all_deltas else 0,
                "max_delta": float(np.max(all_deltas)) if all_deltas else 0,
                "n_edges": len(all_deltas),
                "top_hub_fan_outs": top_hubs,
            })
        model_stats[model] = stats

    if not has_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): Mean |delta| effect size per layer pair
    all_pairs_set = set()
    for stats in model_stats.values():
        for s in stats:
            all_pairs_set.add(s["pair_key"])

    for model in models:
        if model not in model_stats:
            continue
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        stats = model_stats[model]
        pairs = [s["pair_key"] for s in stats]
        means = [s["mean_delta"] for s in stats]
        medians = [s["median_delta"] for s in stats]
        x = np.arange(len(pairs))
        axes[0].bar(x - 0.15, means, 0.3, color=color, alpha=0.7, label=f"{label} mean")
        axes[0].bar(x + 0.15, medians, 0.3, color=color, alpha=0.35,
                    label=f"{label} median", hatch="//")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pairs, fontsize=8)

    axes[0].set_ylabel("|Δ activation| (SAE units)")
    axes[0].set_title("Connection Strength")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel (b): Effect size distributions (box plots)
    all_data = []
    all_labels = []
    all_colors = []
    for model in models:
        if model not in model_stats:
            continue
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        for s in model_stats[model]:
            all_data.append(s["deltas"])
            all_labels.append(f"{label}\n{s['pair_key']}")
            all_colors.append(color)

    if all_data:
        bp = axes[1].boxplot(all_data, labels=all_labels, patch_artist=True,
                             showfliers=False, whis=[5, 95])
        for patch, c in zip(bp["boxes"], all_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        axes[1].tick_params(axis='x', labelsize=7)

    axes[1].set_ylabel("|Δ activation|")
    axes[1].set_title("Effect Size Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Panel (c): Hub connectivity — top-10 upstream features by fan-out
    for model in models:
        if model not in model_stats:
            continue
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        for s in model_stats[model]:
            fan_outs = s["top_hub_fan_outs"]
            axes[2].plot(range(1, len(fan_outs) + 1), fan_outs, "o-",
                        color=color, alpha=0.7, markersize=4,
                        label=f"{label} {s['pair_key']}")

    axes[2].set_xlabel("Hub Rank")
    axes[2].set_ylabel("Fan-Out (# downstream targets)")
    axes[2].set_title("Upstream Hub Features")
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Sparse Feature Circuits", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_sparse_feature_circuits")
    plt.close()


# ============================================================
# Figure 18: Cross-Modal SAE Features
# ============================================================

def fig_crossmodal_sae(results_dict, models, fig_num=18):
    """Cross-modal SAE feature modulation: structure-enhanced/suppressed features."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Modulation fraction by layer for ESM-3
    ax = axes[0]
    for model in models:
        if model not in results_dict or "crossmodal_sae_features" not in results_dict[model]:
            continue
        data = results_dict[model]["crossmodal_sae_features"]
        per_layer = data.get("per_layer", {})
        if not per_layer:
            continue

        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)

        layers = sorted(per_layer.keys())
        fracs = [per_layer[l]["fraction_modulated"] for l in layers]
        ax.bar([f"{l}" for l in layers], fracs,
               color=color, alpha=0.7, label=label)

    ax.set_ylabel("Fraction of Features Modulated")
    ax.set_xlabel("SAE (Layer_Condition)")
    ax.set_title("Structure-Modulated SAE Features")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    # Panel (b): Enhanced vs suppressed breakdown for ESM-3
    ax = axes[1]
    if "esm3" in results_dict and "crossmodal_sae_features" in results_dict.get("esm3", {}):
        data = results_dict["esm3"]["crossmodal_sae_features"]
        per_layer = data.get("per_layer", {})
        layers = sorted(per_layer.keys())
        enhanced = [per_layer[l]["n_structure_enhanced"] for l in layers]
        suppressed = [per_layer[l]["n_structure_suppressed"] for l in layers]
        x = np.arange(len(layers))
        w = 0.35
        ax.bar(x - w/2, enhanced, w, color="#4CAF50", label="Enhanced", alpha=0.8)
        ax.bar(x + w/2, suppressed, w, color="#F44336", label="Suppressed", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, fontsize=8)
        ax.set_ylabel("Number of Features")
        ax.set_xlabel("SAE (Layer_Condition)")
        ax.set_title("ESM-3: Enhanced vs Suppressed by Structure")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Cross-Modal SAE Feature Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_crossmodal_sae")
    plt.close()


# ============================================================
# Figure 19: Evolutionary Correlation
# ============================================================

def fig_evolutionary_correlation(results_dict, models, fig_num=19):
    """Evolutionary conservation correlation: MLM entropy vs SAE/mutation KL."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): Aggregate correlations bar chart
    ax = axes[0]
    metrics = ["entropy_vs_sae", "entropy_vs_mutation_kl"]
    metric_labels = ["Entropy vs SAE", "Entropy vs Mut. KL"]
    x = np.arange(len(metrics))
    w = 0.3
    for i, model in enumerate(models):
        if model not in results_dict or "evolutionary_correlation" not in results_dict[model]:
            continue
        data = results_dict[model]["evolutionary_correlation"]
        agg = data.get("aggregate", {})
        vals = []
        errs = []
        for m in metrics:
            if m in agg:
                vals.append(agg[m].get("mean", 0))
                errs.append(agg[m].get("std", 0))
            else:
                vals.append(0)
                errs.append(0)
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.bar(x + i * w - w/2, vals, w, yerr=errs, color=color,
               label=label, alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Aggregate Correlations")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel (b): Per-protein distribution
    ax = axes[1]
    for model in models:
        if model not in results_dict or "evolutionary_correlation" not in results_dict[model]:
            continue
        data = results_dict[model]["evolutionary_correlation"]
        pp = data.get("per_protein", {})
        rhos = [v["entropy_vs_sae"]["spearman_rho"] for v in pp.values()
                if "entropy_vs_sae" in v and not np.isnan(v["entropy_vs_sae"]["spearman_rho"])]
        if rhos:
            color = MODEL_COLORS.get(model, "gray")
            label = MODEL_LABELS.get(model, model)
            ax.hist(rhos, bins=25, color=color, alpha=0.5, label=label, density=True)
            ax.axvline(np.mean(rhos), color=color, linestyle="--", linewidth=2)

    ax.set_xlabel("Spearman ρ (entropy vs SAE)")
    ax.set_ylabel("Density")
    ax.set_title("Per-Protein Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (c): Breakdown by residue category
    ax = axes[2]
    categories = ["buried", "exposed", "functional", "nonfunctional"]
    cat_labels = ["Buried", "Exposed", "Functional", "Non-func."]
    x = np.arange(len(categories))
    w = 0.3
    for i, model in enumerate(models):
        if model not in results_dict or "evolutionary_correlation" not in results_dict[model]:
            continue
        data = results_dict[model]["evolutionary_correlation"]
        bd = data.get("breakdown", {})
        vals = [bd.get(c, {}).get("entropy_vs_sae_spearman", 0) for c in categories]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.bar(x + i * w - w/2, vals, w, color=color, label=label, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=10)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("By Residue Category")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Evolutionary Conservation Correlation", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_evolutionary_correlation")
    plt.close()


# ============================================================
# Figure 20: DMS Correlation
# ============================================================

def fig_dms_correlation(results_dict, models, fig_num=20):
    """DMS fitness correlation: model importance vs experimental mutation effects."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): Per-protein Spearman ρ for mutation KL
    ax = axes[0]
    for model in models:
        if model not in results_dict or "dms_correlation" not in results_dict[model]:
            continue
        data = results_dict[model]["dms_correlation"]
        pp = data.get("per_protein", {})
        names = sorted(pp.keys())
        rhos = [pp[n]["correlations"]["mutation_kl"]["spearman_rho"] for n in names
                if "mutation_kl" in pp[n].get("correlations", {})]
        pnames = [n for n in names if "mutation_kl" in pp[n].get("correlations", {})]
        if rhos:
            color = MODEL_COLORS.get(model, "gray")
            label = MODEL_LABELS.get(model, model)
            x_pos = np.arange(len(rhos))
            ax.bar(x_pos + (0.2 if model == "esm3" else -0.2), rhos,
                   0.35, color=color, label=label, alpha=0.8)
            if model == models[0]:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(pnames, rotation=45, ha="right", fontsize=7)

    ax.set_ylabel("Spearman ρ")
    ax.set_title("Mutation KL vs DMS Fitness")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel (b): Aggregate across metrics
    ax = axes[1]
    metrics = ["mutation_kl", "sae_activation", "mlm_entropy"]
    metric_labels = ["Mut. KL", "SAE Act.", "MLM Ent."]
    x = np.arange(len(metrics))
    w = 0.3
    for i, model in enumerate(models):
        if model not in results_dict or "dms_correlation" not in results_dict[model]:
            continue
        data = results_dict[model]["dms_correlation"]
        agg = data.get("aggregate", {})
        vals = [agg.get(m, {}).get("mean_spearman", 0) for m in metrics]
        errs = [agg.get(m, {}).get("std_spearman", 0) for m in metrics]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.bar(x + i * w - w/2, vals, w, yerr=errs, color=color,
               label=label, alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Mean Spearman ρ")
    ax.set_title("Aggregate DMS Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Panel (c): Per-protein scatter — mutation KL vs DMS correlation for both models
    ax = axes[2]
    for model in models:
        if model not in results_dict or "dms_correlation" not in results_dict[model]:
            continue
        data = results_dict[model]["dms_correlation"]
        pp = data.get("per_protein", {})
        n_pos = [pp[n]["n_dms_positions"] for n in pp]
        rhos = [pp[n]["correlations"]["mutation_kl"]["spearman_rho"] for n in pp
                if "mutation_kl" in pp[n].get("correlations", {})]
        n_pos = [pp[n]["n_dms_positions"] for n in pp
                 if "mutation_kl" in pp[n].get("correlations", {})]
        if rhos:
            color = MODEL_COLORS.get(model, "gray")
            label = MODEL_LABELS.get(model, model)
            ax.scatter(n_pos, rhos, color=color, label=label, alpha=0.7, s=60)

    ax.set_xlabel("# DMS Positions")
    ax.set_ylabel("Spearman ρ (Mut. KL vs DMS)")
    ax.set_title("Correlation vs Dataset Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Deep Mutational Scanning Correlation", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_dms_correlation")
    plt.close()


# ============================================================
# Figure 21: Phylogenetic Recapitulation
# ============================================================

def fig_phylogenetic_recapitulation(results_dict, models, fig_num=21):
    """Phylogenetic recapitulation: Mantel correlation by layer."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): Mantel ρ by layer
    ax = axes[0]
    for model in models:
        if model not in results_dict or "phylogenetic_recapitulation" not in results_dict[model]:
            continue
        data = results_dict[model]["phylogenetic_recapitulation"]
        agg = data.get("aggregate_by_layer", {})
        if not agg:
            continue
        n_layers = data.get("n_layers", 33)
        layers = sorted(int(l) for l in agg.keys())
        rhos = [agg[str(l)]["mean_mantel_rho"] for l in layers]
        stds = [agg[str(l)]["std_mantel_rho"] for l in layers]
        layer_fracs = [l / (n_layers - 1) for l in layers]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.errorbar(layer_fracs, rhos, yerr=stds, fmt="o-",
                    color=color, label=label, markersize=5,
                    linewidth=1.5, capsize=3)

    ax.set_xlabel("Relative Layer Position")
    ax.set_ylabel("Mean Mantel ρ")
    ax.set_title("Evolutionary Signal by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel (b): Per-family best Mantel ρ
    ax = axes[1]
    for model in models:
        if model not in results_dict or "phylogenetic_recapitulation" not in results_dict[model]:
            continue
        data = results_dict[model]["phylogenetic_recapitulation"]
        pf = data.get("per_family", {})
        if not pf:
            continue
        families = sorted(pf.keys())
        best_rhos = [pf[f]["best_mantel_rho"] for f in families]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.bar(range(len(families)), best_rhos, color=color, alpha=0.6, label=label)
        ax.set_xticks(range(len(families)))
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=7)

    ax.set_ylabel("Best Mantel ρ")
    ax.set_title("Per-Family Best Correlation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Phylogenetic Recapitulation", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_phylogenetic_recapitulation")
    plt.close()


# ============================================================
# Figure 22: Novel Feature Characterization
# ============================================================

def fig_novel_features(results_dict, models, fig_num=22):
    """Novel SAE feature characterization summary."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel (a): Feature classification pie chart (ESM-2 or first available)
    ax = axes[0]
    for model in models:
        if model not in results_dict or "novel_feature_characterization" not in results_dict[model]:
            continue
        data = results_dict[model]["novel_feature_characterization"]
        classified = data.get("summary", {}).get("classified", {})
        if not classified:
            continue
        # Group small categories
        groups = {}
        for cls, cnt in classified.items():
            if cls.startswith("aa_"):
                groups["AA-specific"] = groups.get("AA-specific", 0) + cnt
            elif cls.startswith("ss_"):
                groups["SS-specific"] = groups.get("SS-specific", 0) + cnt
            elif cls == "functional":
                groups["Functional"] = cnt
            elif cls == "dead":
                groups["Dead"] = cnt
            elif cls == "uncharacterized":
                groups["Uncharacterized"] = cnt
            else:
                groups["Other"] = groups.get("Other", 0) + cnt

        labels = list(groups.keys())
        sizes = list(groups.values())
        colors = ["#4CAF50", "#2196F3", "#FF9800", "#9E9E9E", "#F44336", "#9C27B0"]
        ax.pie(sizes, labels=labels, colors=colors[:len(labels)],
               autopct='%1.1f%%', startangle=90)
        ax.set_title(f"{MODEL_LABELS.get(model, model)}\nFeature Classification")
        break  # only one pie chart

    # Panel (b): Hypothesis distribution for both models
    ax = axes[1]
    all_hyps = set()
    for model in models:
        if model not in results_dict or "novel_feature_characterization" not in results_dict[model]:
            continue
        data = results_dict[model]["novel_feature_characterization"]
        hyps = data.get("summary", {}).get("hypothesis_counts", {})
        all_hyps.update(hyps.keys())

    hyp_list = sorted(all_hyps)
    x = np.arange(len(hyp_list))
    w = 0.3
    for i, model in enumerate(models):
        if model not in results_dict or "novel_feature_characterization" not in results_dict[model]:
            continue
        data = results_dict[model]["novel_feature_characterization"]
        hyps = data.get("summary", {}).get("hypothesis_counts", {})
        vals = [hyps.get(h, 0) for h in hyp_list]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.barh(x + i * w - w/2, vals, w, color=color, label=label, alpha=0.8)

    ax.set_yticks(x)
    ax.set_yticklabels([h.replace("_", " ") for h in hyp_list], fontsize=8)
    ax.set_xlabel("Number of Features")
    ax.set_title("Novel Feature Hypotheses")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")

    # Panel (c): Physicochemical profiles of novel features
    ax = axes[2]
    for model in models:
        if model not in results_dict or "novel_feature_characterization" not in results_dict[model]:
            continue
        data = results_dict[model]["novel_feature_characterization"]
        nfs = data.get("novel_features", [])
        if not nfs:
            continue
        hydros = [nf["mean_hydrophobicity"] for nf in nfs]
        asas = [nf["mean_asa"] for nf in nfs]
        color = MODEL_COLORS.get(model, "gray")
        label = MODEL_LABELS.get(model, model)
        ax.scatter(hydros, asas, color=color, alpha=0.4, s=20, label=label)

    ax.set_xlabel("Mean Hydrophobicity")
    ax.set_ylabel("Mean ASA")
    ax.set_title("Novel Feature Profiles")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Novel Feature Characterization", fontsize=14, y=1.02)
    fig.tight_layout()
    save_figure(fig, f"fig{fig_num}_novel_features")
    plt.close()


# ============================================================
# Utilities
# ============================================================

def save_figure(fig, name):
    for ext in ["png", "pdf"]:
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["esm2", "esm3"])
    args = parser.parse_args()

    print(f"=== Generating comparison figures for {args.models} ===")

    results_dict = {}
    for model in args.models:
        results_dict[model] = load_model_results(model)
        print(f"  {model}: {list(results_dict[model].keys())}")

    fig_layer_ablation(results_dict, args.models, fig_num=1)
    fig_head_ablation(results_dict, args.models, fig_num=2)
    fig_attention_properties(results_dict, args.models, fig_num=3)
    fig_activation_patching(results_dict, args.models, fig_num=4)
    fig_cka_within_model(results_dict, args.models, fig_num=5)
    fig_sae_feature_ablation(results_dict, args.models, fig_num=6)
    fig_logit_lens(results_dict, args.models, fig_num=7)
    fig_tuned_lens(results_dict, args.models, fig_num=8)
    fig_direct_logit_attribution(results_dict, args.models, fig_num=9)
    fig_residual_decomposition(results_dict, args.models, fig_num=10)
    fig_ov_qk_decomposition(results_dict, args.models, fig_num=11)
    fig_attribution_patching(results_dict, args.models, fig_num=12)
    fig_mutation_sensitivity(results_dict, args.models, fig_num=13)
    fig_causal_probing(results_dict, args.models, fig_num=14)
    fig_contact_map(results_dict, args.models, fig_num=15)
    fig_steering_vectors(results_dict, args.models, fig_num=16)
    fig_sparse_feature_circuits(results_dict, args.models, fig_num=17)
    fig_crossmodal_sae(results_dict, args.models, fig_num=18)
    fig_evolutionary_correlation(results_dict, args.models, fig_num=19)
    fig_dms_correlation(results_dict, args.models, fig_num=20)
    fig_phylogenetic_recapitulation(results_dict, args.models, fig_num=21)
    fig_novel_features(results_dict, args.models, fig_num=22)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
