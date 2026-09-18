#!/usr/bin/env python3
"""SAE decoder geometry analysis.

Loads all 6 SAE models on CPU, extracts decoder weight vectors,
performs PCA and clustering of the learned feature directions,
colored by biological annotations (SS, AA enrichment, functional).

Reveals the geometric structure of the learned feature space.
"""

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY/src")))
from sae.model import TopKSAE, SAEConfig

BASE = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY")
MODEL_DIR = BASE / "models" / "sae" / "esm3"
RESULTS_DIR = BASE / "results" / "phase1" / "decoder_geometry"
FIG_DIR = BASE / "results" / "figures" / "phase1"

AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Physicochemical AA groups
AA_GROUPS = {
    "hydrophobic": set("AILMFWVP"),
    "polar": set("STNQ"),
    "charged_pos": set("RKH"),
    "charged_neg": set("DE"),
    "aromatic": set("FWY"),
    "small": set("GASP"),
    "cysteine": set("C"),
}

TARGETS = [
    ("S", 16), ("S", 33), ("S", 42),
    ("S+St", 16), ("S+St", 33), ("S+St", 42),
]


def load_sae_decoder(condition: str, layer: int) -> np.ndarray:
    """Load SAE and extract decoder weight matrix."""
    cond_key = "S" if condition == "S" else "S_St"
    model_path = MODEL_DIR / f"{cond_key}_layer_{layer}_topk" / "final.pt"

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # decoder.weight shape: (input_dim, dict_size) = (1536, 12288)
    # Each column is a feature's direction in embedding space
    decoder_weight = state_dict["decoder.weight"].numpy()  # (1536, 12288)
    return decoder_weight.T  # (12288, 1536) — one row per feature


def load_annotations(condition: str, layer: int) -> tuple:
    """Load feature annotations and structural annotations."""
    cond_key = "S" if condition == "S" else "S_St"

    ann_path = BASE / "results" / "phase1" / "modality_saes" / f"{cond_key}_layer_{layer}_annotations.json"
    struct_path = BASE / "results" / "phase1" / "modality_saes" / "structural_annotations" / f"{cond_key}_layer_{layer}_structural.json"

    with open(ann_path) as f:
        ann = json.load(f)
    if isinstance(ann, dict) and "features" in ann:
        ann = ann["features"]

    struct = []
    try:
        with open(struct_path) as f:
            struct_data = json.load(f)
        struct = struct_data["features"] if isinstance(struct_data, dict) else struct_data
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return ann, struct


def classify_feature(ann_entry: dict, struct_entry: dict = None) -> dict:
    """Classify a single feature by its biological properties."""
    # AA enrichment
    enrichments = ann_entry.get("top_enriched_aa", [])
    max_aa_enrich = max((e for _, e in enrichments), default=1.0)
    top_aa = enrichments[0][0] if enrichments and enrichments[0][1] > 2.0 else None

    # Determine AA group
    aa_group = "none"
    if top_aa:
        for group, aas in AA_GROUPS.items():
            if top_aa in aas:
                aa_group = group
                break

    # Functional enrichment
    func_enrich = ann_entry.get("functional_site_enrichment", 1.0)

    # Structural properties
    ss_pref = "none"
    rsa_corr = 0.0
    if struct_entry:
        ss_pref = struct_entry.get("ss_preference", "none")
        rsa_corr = struct_entry.get("rsa_correlation", 0.0)

    # Overall type
    if max_aa_enrich > 3.0:
        ftype = "aa_specific"
    elif func_enrich > 2.0:
        ftype = "functional"
    elif struct_entry and struct_entry.get("max_ss_enrichment", 1.0) > 2.0:
        ftype = "structural"
    else:
        ftype = "generic"

    return {
        "type": ftype,
        "top_aa": top_aa,
        "aa_group": aa_group,
        "max_aa_enrichment": max_aa_enrich,
        "functional_enrichment": func_enrich,
        "ss_preference": ss_pref,
        "rsa_correlation": rsa_corr,
    }


def analyze_decoder(condition: str, layer: int) -> dict:
    """Full decoder geometry analysis for one SAE."""
    print(f"\n  Loading decoder weights...")
    decoder = load_sae_decoder(condition, layer)
    n_features, d_model = decoder.shape
    print(f"  Decoder shape: {decoder.shape}")

    # Load annotations
    ann, struct = load_annotations(condition, layer)

    # Build annotation index
    ann_by_idx = {a["feature_idx"]: a for a in ann}
    struct_by_idx = {s["feature_idx"]: s for s in struct}

    # Identify alive features (non-zero decoder norms)
    norms = np.linalg.norm(decoder, axis=1)
    alive_mask = norms > 0.01
    n_alive = alive_mask.sum()
    print(f"  Alive features: {n_alive}/{n_features}")

    # Use annotated features (top 200)
    annotated_indices = [a["feature_idx"] for a in ann]
    annotated_decoder = decoder[annotated_indices]  # (200, 1536)

    # Normalize decoder vectors
    ann_norms = np.linalg.norm(annotated_decoder, axis=1, keepdims=True)
    ann_norms[ann_norms == 0] = 1
    annotated_normed = annotated_decoder / ann_norms

    # PCA on annotated decoder vectors
    print(f"  Running PCA...")
    pca = PCA(n_components=min(50, len(annotated_indices)))
    pca_coords = pca.fit_transform(annotated_normed)
    explained = pca.explained_variance_ratio_

    print(f"  PCA variance explained: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}, "
          f"PC3={explained[2]:.3f}, top10={sum(explained[:10]):.3f}")

    # t-SNE for visualization
    print(f"  Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    tsne_coords = tsne.fit_transform(annotated_normed)

    # Classify features
    classifications = []
    for idx in annotated_indices:
        ann_entry = ann_by_idx.get(idx, {})
        struct_entry = struct_by_idx.get(idx, None)
        classifications.append(classify_feature(ann_entry, struct_entry))

    # Decoder cosine similarity statistics
    cos_sim = annotated_normed @ annotated_normed.T
    np.fill_diagonal(cos_sim, 0)
    mean_cos = cos_sim.mean()
    max_cos = cos_sim.max()

    # Find most similar feature pairs
    n = len(annotated_indices)
    upper_tri = np.triu_indices(n, k=1)
    pair_sims = cos_sim[upper_tri]
    top_pairs_idx = np.argsort(-pair_sims)[:10]
    top_pairs = []
    for pi in top_pairs_idx:
        i, j = upper_tri[0][pi], upper_tri[1][pi]
        top_pairs.append({
            "feature_i": annotated_indices[i],
            "feature_j": annotated_indices[j],
            "cosine_similarity": float(pair_sims[pi]),
            "type_i": classifications[i]["type"],
            "type_j": classifications[j]["type"],
        })

    # Decoder norm statistics
    ann_norm_values = norms[annotated_indices]

    return {
        "condition": condition,
        "layer": layer,
        "n_features": n_features,
        "n_alive": int(n_alive),
        "n_annotated": len(annotated_indices),
        "pca_variance_explained": explained[:10].tolist(),
        "pca_coords": pca_coords[:, :3].tolist(),  # Top 3 PCs
        "tsne_coords": tsne_coords.tolist(),
        "classifications": classifications,
        "feature_indices": annotated_indices,
        "decoder_norms": ann_norm_values.tolist(),
        "cosine_similarity_stats": {
            "mean": float(mean_cos),
            "max": float(max_cos),
            "std": float(cos_sim[upper_tri].std()),
        },
        "top_similar_pairs": top_pairs,
    }


def generate_figures(all_results: dict):
    """Generate decoder geometry visualization figures."""

    # Figure 15: t-SNE of decoder vectors for 3 key SAEs, colored by feature type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    key_saes = [("S", 33), ("S", 42), ("S+St", 33)]
    type_colors = {
        "aa_specific": "#E91E63",
        "functional": "#FF9800",
        "structural": "#4CAF50",
        "generic": "#9E9E9E",
    }

    for ax_idx, (cond, layer) in enumerate(key_saes):
        key = f"{cond}_layer_{layer}"
        data = all_results[key]
        coords = np.array(data["tsne_coords"])
        cls = data["classifications"]

        ax = axes[ax_idx]
        for ftype, color in type_colors.items():
            mask = [c["type"] == ftype for c in cls]
            if any(mask):
                idx = np.where(mask)[0]
                ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=20,
                           alpha=0.7, label=ftype.replace("_", " "))

        ax.set_title(f'{cond} Layer {layer}', fontsize=13, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel('t-SNE 2', fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=9, markerscale=2)

    fig.suptitle('SAE Decoder Vector Geometry (t-SNE)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig15_decoder_tsne.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 16: t-SNE colored by SS preference and RSA correlation
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ss_colors = {"H": "#E91E63", "E": "#2196F3", "C": "#4CAF50", "none": "#BDBDBD"}

    for col, (cond, layer) in enumerate(key_saes):
        key = f"{cond}_layer_{layer}"
        data = all_results[key]
        coords = np.array(data["tsne_coords"])
        cls = data["classifications"]

        # Row 0: SS preference
        ax = axes[0, col]
        for ss, color in ss_colors.items():
            mask = [c["ss_preference"] == ss for c in cls]
            if any(mask):
                idx = np.where(mask)[0]
                ss_label = {"H": "Helix", "E": "Sheet", "C": "Coil", "none": "None"}.get(ss, ss)
                ax.scatter(coords[idx, 0], coords[idx, 1], c=color, s=20,
                           alpha=0.7, label=ss_label)
        ax.set_title(f'{cond} L{layer}', fontsize=12, fontweight='bold')
        if col == 0:
            ax.set_ylabel('SS Preference\nt-SNE 2', fontsize=11)
            ax.legend(fontsize=8, markerscale=2)

        # Row 1: RSA correlation
        ax = axes[1, col]
        rsa_vals = [c["rsa_correlation"] for c in cls]
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=rsa_vals, cmap='RdBu_r',
                        vmin=-0.3, vmax=0.3, s=20, alpha=0.7)
        if col == 2:
            plt.colorbar(sc, ax=ax, label='RSA correlation', shrink=0.8)
        if col == 0:
            ax.set_ylabel('RSA Correlation\nt-SNE 2', fontsize=11)
        ax.set_xlabel('t-SNE 1', fontsize=10)

    fig.suptitle('SAE Feature Space: Structural Properties', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig16_decoder_structural.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 17: PCA variance explained comparison + decoder norm distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: PCA variance explained
    ax = axes[0]
    for cond, layer in TARGETS:
        key = f"{cond}_layer_{layer}"
        data = all_results[key]
        var_exp = data["pca_variance_explained"]
        cumvar = np.cumsum(var_exp)
        style = '-' if cond == "S" else '--'
        colors = {16: '#2196F3', 33: '#4CAF50', 42: '#FF5722'}
        ax.plot(range(1, len(cumvar) + 1), cumvar, style, linewidth=2,
                color=colors[layer], label=f'{cond} L{layer}')

    ax.set_xlabel('Number of PCs', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained', fontsize=12)
    ax.set_title('A. Decoder Dimensionality', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(1, 10)
    ax.grid(alpha=0.3)

    # Panel B: Cosine similarity stats
    ax = axes[1]
    conditions = []
    means = []
    maxes = []

    for cond, layer in TARGETS:
        key = f"{cond}_layer_{layer}"
        data = all_results[key]
        cs = data["cosine_similarity_stats"]
        conditions.append(f"{cond}\nL{layer}")
        means.append(cs["mean"])
        maxes.append(cs["max"])

    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, means, width, label='Mean cos sim', color='#2196F3', alpha=0.85)
    ax.bar(x + width/2, maxes, width, label='Max cos sim', color='#FF5722', alpha=0.85)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('B. Feature Direction Similarity', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.legend(fontsize=10)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig17_decoder_pca.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved: fig15_decoder_tsne, fig16_decoder_structural, fig17_decoder_pca")


def main():
    print("=" * 70)
    print("SAE Decoder Geometry Analysis")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for condition, layer in TARGETS:
        print(f"\n{'='*50}")
        print(f"  {condition} Layer {layer}")
        print(f"{'='*50}")

        result = analyze_decoder(condition, layer)
        key = f"{condition}_layer_{layer}"
        all_results[key] = result

    # Cross-condition comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'SAE':>12} | {'Alive':>6} | {'PCA top3':>10} | {'Mean cos':>10} | {'Max cos':>10}")
    print("-" * 60)
    for cond, layer in TARGETS:
        key = f"{cond}_layer_{layer}"
        d = all_results[key]
        var3 = sum(d["pca_variance_explained"][:3])
        cs = d["cosine_similarity_stats"]
        print(f"{cond+' L'+str(layer):>12} | {d['n_alive']:>6} | {var3:>10.3f} | "
              f"{cs['mean']:>10.4f} | {cs['max']:>10.4f}")

    # Count feature types per SAE
    print(f"\n{'SAE':>12} | {'AA-spec':>8} | {'Func':>8} | {'Struct':>8} | {'Generic':>8}")
    print("-" * 58)
    for cond, layer in TARGETS:
        key = f"{cond}_layer_{layer}"
        d = all_results[key]
        type_counts = defaultdict(int)
        for c in d["classifications"]:
            type_counts[c["type"]] += 1
        print(f"{cond+' L'+str(layer):>12} | {type_counts['aa_specific']:>8} | "
              f"{type_counts['functional']:>8} | {type_counts['structural']:>8} | "
              f"{type_counts['generic']:>8}")

    # Save results (exclude large arrays from JSON)
    save_results = {}
    for key, data in all_results.items():
        save_data = {k: v for k, v in data.items()
                     if k not in ("pca_coords", "tsne_coords")}
        # Save coords separately as compact arrays
        save_data["pca_coords_shape"] = np.array(data["pca_coords"]).shape
        save_data["tsne_coords_shape"] = np.array(data["tsne_coords"]).shape
        save_results[key] = save_data

    out_path = RESULTS_DIR / "decoder_geometry.json"
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Save full coords as numpy for later use
    for key, data in all_results.items():
        np.savez(RESULTS_DIR / f"{key.replace('+', '_plus_')}_coords.npz",
                 pca=np.array(data["pca_coords"]),
                 tsne=np.array(data["tsne_coords"]))

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
