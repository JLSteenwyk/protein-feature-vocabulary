#!/usr/bin/env python3
"""Analysis 3: SAE feature co-activation analysis.

Loads trained SAE models on CPU, runs forward pass on saved HDF5 activations,
computes feature correlation matrix from sparse activation vectors (z),
and clusters features into interpretable modules.
"""

import json
import h5py
import torch
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY/src")))
from sae.model import TopKSAE, SAEConfig

BASE = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY")
ACT_DIR = BASE / "data" / "activations" / "esm3_multimodal"
MODEL_DIR = BASE / "models" / "sae" / "esm3"
RESULTS_DIR = BASE / "results" / "phase1" / "coactivation"
FIG_DIR = BASE / "results" / "figures" / "phase1"

# Standard amino acids
AAS = list("ACDEFGHIKLMNPQRSTVWY")

# Analyze the richest SAEs
TARGETS = [
    ("S", 33),     # Richest in functional features
    ("S", 42),     # Richest in structural features
    ("S+St", 33),  # For comparison
]

BATCH_SIZE = 4096  # Process activations in batches for memory efficiency
MAX_FEATURES = 2000  # Top features by activation frequency to analyze


def load_sae(condition: str, layer: int) -> TopKSAE:
    """Load trained SAE on CPU."""
    cond_key = "S" if condition == "S" else "S_St"
    model_path = MODEL_DIR / f"{cond_key}_layer_{layer}_topk" / "final.pt"

    # Load checkpoint (saved as dict with model_state_dict + sae_config)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    # Extract config and state dict
    config = ckpt["sae_config"]  # SAEConfig dataclass
    state_dict = ckpt["model_state_dict"]

    model = TopKSAE(config)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def compute_activations(model: TopKSAE, h5_path: str, max_residues: int = 100000) -> np.ndarray:
    """Run SAE encoder on saved activations, return sparse activation matrix.

    Returns: (n_residues, dict_size) sparse activation matrix
    """
    with h5py.File(h5_path, "r") as hf:
        total = hf["activations"].shape[0]
        n = min(total, max_residues)
        dim = hf["activations"].shape[1]
        dict_size = model.config.dict_size

        print(f"    Processing {n}/{total} residues (dim={dim}, dict_size={dict_size})")

        # Collect sparse activations
        # We'll store feature indices and values for memory efficiency
        all_active_features = []  # list of (feature_idx,) arrays per residue

        # Process in batches
        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = torch.tensor(hf["activations"][start:end], dtype=torch.float32)

            with torch.no_grad():
                z = model.encode(batch)  # (batch, dict_size)

            # Store as scipy sparse or just active indices
            z_np = z.numpy()
            all_active_features.append(z_np)

            if start % (BATCH_SIZE * 10) == 0:
                print(f"      Batch {start//BATCH_SIZE + 1}/{(n + BATCH_SIZE - 1)//BATCH_SIZE}")

    return np.concatenate(all_active_features, axis=0)


def compute_coactivation(z_matrix: np.ndarray, top_k: int = MAX_FEATURES) -> dict:
    """Compute co-activation statistics from sparse activation matrix.

    Args:
        z_matrix: (n_residues, dict_size) array
        top_k: number of top features to analyze

    Returns:
        dict with correlation matrix, feature stats, etc.
    """
    n_residues, dict_size = z_matrix.shape

    # Find most frequently active features
    active_mask = z_matrix > 0  # (n_residues, dict_size)
    feature_freq = active_mask.mean(axis=0)  # (dict_size,)

    # Select top features by frequency (excluding dead features)
    alive = feature_freq > 0.001  # at least 0.1% activation rate
    n_alive = alive.sum()
    print(f"    Alive features (>0.1% activation): {n_alive}/{dict_size}")

    # Get top-k alive features
    alive_indices = np.where(alive)[0]
    if len(alive_indices) > top_k:
        freq_order = np.argsort(-feature_freq[alive_indices])
        top_indices = alive_indices[freq_order[:top_k]]
    else:
        top_indices = alive_indices

    print(f"    Analyzing top {len(top_indices)} features")

    # Extract sub-matrix for top features
    z_sub = z_matrix[:, top_indices]  # (n_residues, top_k)

    # Binary co-activation: are features active together?
    binary = (z_sub > 0).astype(np.float32)

    # Correlation of binary activation patterns (phi coefficient)
    # Center the binary matrix
    means = binary.mean(axis=0, keepdims=True)
    centered = binary - means
    stds = binary.std(axis=0, keepdims=True)
    stds[stds == 0] = 1  # avoid division by zero
    normalized = centered / stds

    # Correlation matrix
    n = normalized.shape[0]
    corr = (normalized.T @ normalized) / n  # (top_k, top_k)

    # Also compute magnitude-weighted co-activation
    # Pearson correlation of activation magnitudes (for active features)
    z_means = z_sub.mean(axis=0, keepdims=True)
    z_centered = z_sub - z_means
    z_stds = z_sub.std(axis=0, keepdims=True)
    z_stds[z_stds == 0] = 1
    z_normalized = z_centered / z_stds
    mag_corr = (z_normalized.T @ z_normalized) / n

    return {
        "feature_indices": top_indices.tolist(),
        "feature_frequencies": feature_freq[top_indices].tolist(),
        "binary_correlation": corr,
        "magnitude_correlation": mag_corr,
        "n_alive": int(n_alive),
        "n_analyzed": len(top_indices),
        "n_residues": n_residues,
    }


def cluster_features(corr_matrix: np.ndarray, n_clusters: int = 15) -> np.ndarray:
    """Hierarchical clustering of features based on correlation."""
    # Convert correlation to distance
    dist = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist, 0)

    # Make symmetric and ensure non-negative
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)

    # Condensed distance matrix
    dist_condensed = squareform(dist, checks=False)

    # Hierarchical clustering
    Z = linkage(dist_condensed, method='ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return labels, Z


def annotate_clusters(labels: np.ndarray, feature_indices: list,
                      annotations_path: str, structural_path: str) -> list:
    """Annotate each cluster with biological meaning."""
    # Load annotations
    with open(annotations_path) as f:
        ann_data = json.load(f)
    if isinstance(ann_data, dict) and "features" in ann_data:
        ann_data = ann_data["features"]

    # Index annotations by feature_idx
    ann_by_idx = {}
    for a in ann_data:
        ann_by_idx[a["feature_idx"]] = a

    # Load structural annotations if available
    struct_by_idx = {}
    try:
        with open(structural_path) as f:
            struct_data = json.load(f)
        if isinstance(struct_data, dict) and "features" in struct_data:
            struct_data = struct_data["features"]
        for s in struct_data:
            struct_by_idx[s["feature_idx"]] = s
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    cluster_ids = sorted(set(labels))
    cluster_info = []

    for cid in cluster_ids:
        member_mask = labels == cid
        member_indices = [feature_indices[i] for i in range(len(labels)) if labels[i] == cid]

        # Collect AA enrichments for cluster members
        all_aa_enrichments = defaultdict(list)
        func_enrichments = []
        ss_preferences = []
        rsa_correlations = []

        for fidx in member_indices:
            if fidx in ann_by_idx:
                ann = ann_by_idx[fidx]
                for aa, enr in ann.get("top_enriched_aa", []):
                    if enr > 1.5:
                        all_aa_enrichments[aa].append(enr)
                func_enrichments.append(ann.get("functional_site_enrichment", 1.0))

            if fidx in struct_by_idx:
                s = struct_by_idx[fidx]
                ss_preferences.append(s.get("ss_preference", "?"))
                rsa_correlations.append(s.get("rsa_correlation", 0.0))

        # Determine cluster character
        # Most common enriched AAs
        top_aas = sorted(all_aa_enrichments.items(), key=lambda x: -np.mean(x[1]))[:3]
        mean_func = np.mean(func_enrichments) if func_enrichments else 1.0
        max_func = max(func_enrichments) if func_enrichments else 1.0

        # SS preference
        ss_counts = Counter(ss_preferences)
        dominant_ss = ss_counts.most_common(1)[0] if ss_counts else ("?", 0)
        mean_rsa = np.mean(rsa_correlations) if rsa_correlations else 0.0

        # Auto-label
        label_parts = []
        if top_aas and top_aas[0][1] and np.mean(top_aas[0][1]) > 2.0:
            label_parts.append(f"AA:{','.join(aa for aa, _ in top_aas[:2])}")
        if mean_func > 1.5:
            label_parts.append(f"Func({mean_func:.1f}×)")
        if dominant_ss[0] != "?" and dominant_ss[1] > len(member_indices) * 0.4:
            ss_names = {"H": "Helix", "E": "Sheet", "C": "Coil"}
            label_parts.append(ss_names.get(dominant_ss[0], dominant_ss[0]))
        if abs(mean_rsa) > 0.1:
            label_parts.append("Buried" if mean_rsa < -0.1 else "Surface")

        auto_label = " / ".join(label_parts) if label_parts else "Generic"

        cluster_info.append({
            "cluster_id": int(cid),
            "n_features": int(member_mask.sum()),
            "feature_indices": member_indices,
            "auto_label": auto_label,
            "top_enriched_aa": [(aa, float(np.mean(enrs))) for aa, enrs in top_aas],
            "mean_functional_enrichment": float(mean_func),
            "max_functional_enrichment": float(max_func),
            "dominant_ss": dominant_ss[0],
            "ss_purity": float(dominant_ss[1] / len(member_indices)) if len(member_indices) > 0 else 0,
            "mean_rsa_correlation": float(mean_rsa),
        })

    return cluster_info


def generate_figures(results: dict):
    """Generate co-activation figures for all analyzed SAEs."""

    for target_key, target_data in results.items():
        if not isinstance(target_data, dict) or "clusters" not in target_data:
            continue

        condition, layer = target_key.split("_layer_")
        layer = int(layer)

        corr = np.array(target_data["correlation_matrix"])
        labels = np.array(target_data["cluster_labels"])
        clusters = target_data["clusters"]
        n_feat = corr.shape[0]

        # Reorder by cluster for visualization
        order = np.argsort(labels)
        corr_ordered = corr[np.ix_(order, order)]
        labels_ordered = labels[order]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Panel A: Correlation matrix reordered by cluster
        ax = axes[0]
        im = ax.imshow(corr_ordered[:200, :200], cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')

        # Add cluster boundaries
        cluster_boundaries = []
        for i in range(1, len(labels_ordered)):
            if labels_ordered[i] != labels_ordered[i-1] and i < 200:
                cluster_boundaries.append(i)
        for b in cluster_boundaries:
            ax.axhline(y=b-0.5, color='black', linewidth=0.5, alpha=0.5)
            ax.axvline(x=b-0.5, color='black', linewidth=0.5, alpha=0.5)

        ax.set_title(f'A. Feature Co-activation ({condition} L{layer})\n(top 200 features, clustered)',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature (reordered)', fontsize=11)
        ax.set_ylabel('Feature (reordered)', fontsize=11)
        plt.colorbar(im, ax=ax, label='Binary correlation (φ)', shrink=0.8)

        # Panel B: Cluster summary
        ax = axes[1]

        # Sort clusters by size
        sorted_clusters = sorted(clusters, key=lambda c: -c["n_features"])[:12]

        y_pos = np.arange(len(sorted_clusters))
        sizes = [c["n_features"] for c in sorted_clusters]
        colors_list = []
        for c in sorted_clusters:
            if c["mean_functional_enrichment"] > 1.5:
                colors_list.append('#FF5722')
            elif c["dominant_ss"] == "H":
                colors_list.append('#4CAF50')
            elif c["dominant_ss"] == "E":
                colors_list.append('#2196F3')
            else:
                colors_list.append('#9E9E9E')

        bars = ax.barh(y_pos, sizes, color=colors_list, alpha=0.85)
        labels_text = [f"C{c['cluster_id']}: {c['auto_label']}" for c in sorted_clusters]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_text, fontsize=9)
        ax.set_xlabel('Number of features', fontsize=11)
        ax.set_title(f'B. Feature Modules ({condition} L{layer})',
                     fontsize=12, fontweight='bold')
        ax.invert_yaxis()

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor='#FF5722', alpha=0.85, label='Functional'),
            mpatches.Patch(facecolor='#4CAF50', alpha=0.85, label='Helix'),
            mpatches.Patch(facecolor='#2196F3', alpha=0.85, label='Sheet'),
            mpatches.Patch(facecolor='#9E9E9E', alpha=0.85, label='Other'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()

        cond_key = condition.replace("+", "")
        for ext in ['png', 'pdf']:
            fig.savefig(FIG_DIR / f'fig12_coactivation_{cond_key}_L{layer}.{ext}',
                       dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Figures saved to {FIG_DIR}/fig12_*")


def main():
    print("=" * 70)
    print("Analysis 3: SAE Feature Co-activation Analysis")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for condition, layer in TARGETS:
        print(f"\n{'='*50}")
        print(f"Processing {condition} Layer {layer}")
        print(f"{'='*50}")

        # Load SAE model
        print(f"\n  Loading SAE model...")
        model = load_sae(condition, layer)
        print(f"  Model loaded: {model.config.dict_size} features, dim={model.config.input_dim}")

        # Load activations and compute SAE features
        h5_path = ACT_DIR / f"{condition}_layer_{layer}.h5"
        print(f"\n  Computing SAE activations from {h5_path.name}...")
        z_matrix = compute_activations(model, str(h5_path), max_residues=80000)
        print(f"  Activation matrix: {z_matrix.shape}")

        # Compute co-activation
        print(f"\n  Computing co-activation statistics...")
        coact = compute_coactivation(z_matrix)
        print(f"  Analyzed {coact['n_analyzed']} features, {coact['n_alive']} alive")

        # Cluster features
        print(f"\n  Clustering features...")
        n_clusters = 15
        labels, linkage_Z = cluster_features(coact["binary_correlation"], n_clusters=n_clusters)
        cluster_sizes = Counter(labels)
        print(f"  {n_clusters} clusters, sizes: {sorted(cluster_sizes.values(), reverse=True)}")

        # Annotate clusters
        print(f"\n  Annotating clusters...")
        cond_key = "S" if condition == "S" else "S_St"
        ann_path = BASE / "results" / "phase1" / "modality_saes" / f"{cond_key}_layer_{layer}_annotations.json"
        struct_path = BASE / "results" / "phase1" / "modality_saes" / "structural_annotations" / f"{cond_key}_layer_{layer}_structural.json"

        cluster_info = annotate_clusters(
            labels, coact["feature_indices"], str(ann_path), str(struct_path)
        )

        print(f"\n  Cluster summary:")
        for ci in sorted(cluster_info, key=lambda x: -x["n_features"]):
            print(f"    C{ci['cluster_id']:>2}: {ci['n_features']:>4} features  |  {ci['auto_label']}")

        # Store results (convert numpy to lists for JSON)
        target_key = f"{condition}_layer_{layer}"
        all_results[target_key] = {
            "condition": condition,
            "layer": layer,
            "n_residues": coact["n_residues"],
            "n_alive_features": coact["n_alive"],
            "n_analyzed_features": coact["n_analyzed"],
            "feature_indices": coact["feature_indices"],
            "feature_frequencies": coact["feature_frequencies"],
            "correlation_matrix": coact["binary_correlation"].tolist(),
            "cluster_labels": labels.tolist(),
            "n_clusters": n_clusters,
            "clusters": cluster_info,
        }

        # Free memory
        del z_matrix, model

    # Summary comparison
    print("\n" + "=" * 70)
    print("CROSS-CONDITION COMPARISON")
    print("=" * 70)

    for target_key, data in all_results.items():
        print(f"\n{target_key}:")
        print(f"  Alive features: {data['n_alive_features']}")

        # Count clusters by dominant type
        type_counts = defaultdict(int)
        for ci in data["clusters"]:
            label = ci["auto_label"]
            if "Func" in label:
                type_counts["Functional"] += 1
            elif "Helix" in label or "Sheet" in label or "Coil" in label:
                type_counts["Structural"] += 1
            elif "AA:" in label:
                type_counts["AA-specific"] += 1
            elif "Buried" in label or "Surface" in label:
                type_counts["Accessibility"] += 1
            else:
                type_counts["Generic"] += 1

        print(f"  Cluster types: {dict(type_counts)}")

    # Save results
    out_path = RESULTS_DIR / "coactivation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(all_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
