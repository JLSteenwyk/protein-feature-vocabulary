#!/usr/bin/env python3
"""Analysis 2: ESM-2 vs ESM-3 SAE feature comparison.

Compares Phase 0 ESM-2 layer 24 SAE features with Phase 1 ESM-3 S-only layer 33
SAE features. These are the closest equivalent layers (both near integration
midpoints). Identifies convergent features (shared AA patterns) and
architecture-specific features.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY")
RESULTS_DIR = BASE / "results" / "phase1" / "cross_model_comparison"
FIG_DIR = BASE / "results" / "figures" / "phase1"

# Standard amino acids
AAS = list("ACDEFGHIKLMNPQRSTVWY")


def load_annotations(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    # Handle both list and dict formats
    if isinstance(data, dict) and "features" in data:
        return data["features"]
    return data


def build_aa_vector(feature: dict) -> np.ndarray:
    """Build 20-dimensional AA enrichment vector from feature annotation."""
    vec = np.ones(20)  # default enrichment = 1.0 (no enrichment)
    for aa, enrichment in feature.get("top_enriched_aa", []):
        if aa in AAS:
            idx = AAS.index(aa)
            vec[idx] = enrichment
    return vec


def build_feature_matrix(annotations: list) -> np.ndarray:
    """Build (n_features, 20) matrix of AA enrichment vectors."""
    vecs = [build_aa_vector(f) for f in annotations]
    return np.array(vecs)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(np.dot(a, b) / norm)


def compute_pairwise_similarity(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two feature matrices."""
    # Normalize rows
    norms1 = np.linalg.norm(mat1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(mat2, axis=1, keepdims=True)
    norms1[norms1 == 0] = 1
    norms2[norms2 == 0] = 1
    mat1_norm = mat1 / norms1
    mat2_norm = mat2 / norms2
    return mat1_norm @ mat2_norm.T


def hungarian_matching(sim_matrix: np.ndarray) -> list:
    """Find optimal 1-to-1 matching using Hungarian algorithm."""
    # linear_sum_assignment minimizes, so negate similarity
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "esm2_idx": int(i),
            "esm3_idx": int(j),
            "similarity": float(sim_matrix[i, j])
        })
    return sorted(matches, key=lambda x: -x["similarity"])


def classify_features(esm2_ann: list, esm3_ann: list) -> dict:
    """Classify features by type (AA-specific, functional, generic)."""
    def classify(feat):
        enrichments = [e for _, e in feat.get("top_enriched_aa", [])]
        max_enrich = max(enrichments) if enrichments else 1.0
        func_enrich = feat.get("functional_site_enrichment", 1.0)

        if max_enrich > 3.0:
            return "aa_specific"
        elif func_enrich > 2.0:
            return "functional"
        else:
            return "generic"

    esm2_types = defaultdict(int)
    esm3_types = defaultdict(int)

    for f in esm2_ann:
        esm2_types[classify(f)] += 1
    for f in esm3_ann:
        esm3_types[classify(f)] += 1

    return {
        "esm2": dict(esm2_types),
        "esm3": dict(esm3_types),
    }


def find_convergent_features(matches: list, esm2_ann: list, esm3_ann: list,
                              threshold: float = 0.95) -> list:
    """Identify convergent features (high similarity) and characterize them."""
    convergent = []
    for m in matches:
        if m["similarity"] >= threshold:
            e2 = esm2_ann[m["esm2_idx"]]
            e3 = esm3_ann[m["esm3_idx"]]

            # Find shared AA enrichments
            e2_aas = {aa for aa, enr in e2.get("top_enriched_aa", []) if enr > 2.0}
            e3_aas = {aa for aa, enr in e3.get("top_enriched_aa", []) if enr > 2.0}
            shared = e2_aas & e3_aas

            convergent.append({
                "esm2_feature": e2["feature_idx"],
                "esm3_feature": e3["feature_idx"],
                "similarity": m["similarity"],
                "esm2_top_aa": e2.get("top_enriched_aa", [])[:5],
                "esm3_top_aa": e3.get("top_enriched_aa", [])[:5],
                "shared_enriched_aa": sorted(shared),
                "esm2_func_enrichment": e2.get("functional_site_enrichment", 1.0),
                "esm3_func_enrichment": e3.get("functional_site_enrichment", 1.0),
            })
    return convergent


def generate_figures(sim_matrix, matches, esm2_ann, esm3_ann, feature_types, results):
    """Generate comparison figures."""

    # Figure 10A: Similarity distribution
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Distribution of best match similarities
    match_sims = [m["similarity"] for m in matches]
    ax = axes[0]
    ax.hist(match_sims, bins=30, color='#2196F3', alpha=0.85, edgecolor='white')
    ax.axvline(x=0.95, color='red', linestyle='--', linewidth=1.5, label='Convergence threshold')
    n_convergent = sum(1 for s in match_sims if s >= 0.95)
    ax.set_xlabel('Cosine Similarity (best match)', fontsize=11)
    ax.set_ylabel('Number of feature pairs', fontsize=11)
    ax.set_title(f'A. Matched Feature Similarity\n({n_convergent}/{len(matches)} convergent)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

    # Panel B: Feature type comparison
    ax = axes[1]
    types = ["aa_specific", "functional", "generic"]
    type_labels = ["AA-specific\n(>3× enrichment)", "Functional\n(>2× enrichment)", "Generic"]
    x = np.arange(len(types))
    width = 0.35

    esm2_counts = [feature_types["esm2"].get(t, 0) for t in types]
    esm3_counts = [feature_types["esm3"].get(t, 0) for t in types]

    ax.bar(x - width/2, esm2_counts, width, label='ESM-2 (L24)', color='#4CAF50', alpha=0.85)
    ax.bar(x + width/2, esm3_counts, width, label='ESM-3 S-only (L33)', color='#FF5722', alpha=0.85)
    ax.set_ylabel('Number of features', fontsize=11)
    ax.set_title('B. Feature Type Distribution', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels, fontsize=9)
    ax.legend(fontsize=10)

    # Panel C: Similarity heatmap (top 50×50 for readability)
    ax = axes[2]
    # Sort by max similarity for visual clarity
    max_sims_rows = sim_matrix.max(axis=1)
    max_sims_cols = sim_matrix.max(axis=0)
    row_order = np.argsort(-max_sims_rows)[:50]
    col_order = np.argsort(-max_sims_cols)[:50]
    sub_matrix = sim_matrix[np.ix_(row_order, col_order)]

    im = ax.imshow(sub_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0.8, vmax=1.0)
    ax.set_xlabel('ESM-3 features (top 50)', fontsize=11)
    ax.set_ylabel('ESM-2 features (top 50)', fontsize=11)
    ax.set_title('C. AA Enrichment Similarity', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Cosine similarity')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig10_esm2_esm3_comparison.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 11: Convergent feature AA profiles
    convergent = results.get("convergent_features", [])
    n_show = min(8, len(convergent))
    if n_show > 0:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()

        for i in range(n_show):
            cf = convergent[i]
            ax = axes[i]

            # Build full AA enrichment vectors for these features
            e2_feat = next(f for f in esm2_ann if f["feature_idx"] == cf["esm2_feature"])
            e3_feat = next(f for f in esm3_ann if f["feature_idx"] == cf["esm3_feature"])

            e2_vec = build_aa_vector(e2_feat)
            e3_vec = build_aa_vector(e3_feat)

            x = np.arange(20)
            width = 0.35
            ax.bar(x - width/2, e2_vec, width, color='#4CAF50', alpha=0.7, label='ESM-2')
            ax.bar(x + width/2, e3_vec, width, color='#FF5722', alpha=0.7, label='ESM-3')
            ax.set_xticks(x)
            ax.set_xticklabels(AAS, fontsize=7)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.5)
            ax.set_title(f'sim={cf["similarity"]:.3f}\nShared: {",".join(cf["shared_enriched_aa"][:3]) or "none"}',
                        fontsize=9)
            if i == 0:
                ax.legend(fontsize=8)
            ax.set_ylim(0, max(e2_vec.max(), e3_vec.max()) * 1.2)

        # Hide unused axes
        for i in range(n_show, 8):
            axes[i].set_visible(False)

        fig.suptitle('Top Convergent Feature Pairs: AA Enrichment Profiles',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(FIG_DIR / f'fig11_convergent_features.{ext}', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Figures saved to {FIG_DIR}/fig10_*, fig11_*")


def main():
    print("=" * 70)
    print("Analysis 2: ESM-2 vs ESM-3 SAE Feature Comparison")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load ESM-2 layer 24 annotations
    print("\nLoading ESM-2 layer 24 SAE annotations...")
    esm2_path = BASE / "results" / "phase0" / "sae_reproduction" / "feature_annotations.json"
    esm2_ann = load_annotations(str(esm2_path))
    print(f"  {len(esm2_ann)} ESM-2 features")

    # Load ESM-3 S-only layer 33 annotations (closest equivalent)
    print("Loading ESM-3 S-only layer 33 SAE annotations...")
    esm3_path = BASE / "results" / "phase1" / "modality_saes" / "S_layer_33_annotations.json"
    esm3_ann = load_annotations(str(esm3_path))
    print(f"  {len(esm3_ann)} ESM-3 features")

    # Also load structural annotations for additional comparison
    esm3_struct_path = BASE / "results" / "phase1" / "modality_saes" / "structural_annotations" / "S_layer_33_structural.json"
    esm3_struct = load_annotations(str(esm3_struct_path))
    print(f"  {len(esm3_struct)} ESM-3 structural annotations")

    # Build AA enrichment matrices
    print("\nBuilding AA enrichment matrices...")
    esm2_mat = build_feature_matrix(esm2_ann)
    esm3_mat = build_feature_matrix(esm3_ann)
    print(f"  ESM-2 matrix: {esm2_mat.shape}")
    print(f"  ESM-3 matrix: {esm3_mat.shape}")

    # Compute pairwise similarity
    print("\nComputing pairwise cosine similarity...")
    sim_matrix = compute_pairwise_similarity(esm2_mat, esm3_mat)
    print(f"  Similarity matrix: {sim_matrix.shape}")
    print(f"  Mean similarity: {sim_matrix.mean():.4f}")
    print(f"  Max similarity: {sim_matrix.max():.4f}")

    # Hungarian matching
    print("\nFinding optimal feature matching (Hungarian algorithm)...")
    matches = hungarian_matching(sim_matrix)
    match_sims = [m["similarity"] for m in matches]
    print(f"  Mean match similarity: {np.mean(match_sims):.4f}")
    print(f"  Median match similarity: {np.median(match_sims):.4f}")

    # Count convergent (>0.95 similarity)
    n_convergent_95 = sum(1 for s in match_sims if s >= 0.95)
    n_convergent_99 = sum(1 for s in match_sims if s >= 0.99)
    n_divergent = sum(1 for s in match_sims if s < 0.90)
    print(f"  Convergent (>0.95): {n_convergent_95}/{len(matches)}")
    print(f"  Highly convergent (>0.99): {n_convergent_99}/{len(matches)}")
    print(f"  Divergent (<0.90): {n_divergent}/{len(matches)}")

    # Permutation test for convergence significance
    print("\nPermutation test (1000 shuffles)...")
    rng = np.random.default_rng(42)
    null_mean_sims = []
    for _ in range(1000):
        # Shuffle ESM-3 AA profiles
        shuffled_mat = esm3_mat.copy()
        rng.shuffle(shuffled_mat, axis=1)  # Shuffle AA assignments
        null_sim = compute_pairwise_similarity(esm2_mat, shuffled_mat)
        null_matches = hungarian_matching(null_sim)
        null_mean_sims.append(np.mean([m["similarity"] for m in null_matches]))

    observed_mean = np.mean(match_sims)
    null_mean = np.mean(null_mean_sims)
    null_std = np.std(null_mean_sims)
    z_score = (observed_mean - null_mean) / null_std if null_std > 0 else float('inf')
    p_val = np.mean([n >= observed_mean for n in null_mean_sims])

    print(f"  Observed mean similarity: {observed_mean:.4f}")
    print(f"  Null mean similarity: {null_mean:.4f} +/- {null_std:.4f}")
    print(f"  Z-score: {z_score:.2f}")
    print(f"  p-value: {p_val:.4f} ({'significant' if p_val < 0.01 else 'not significant'})")

    # Feature type classification
    print("\nClassifying feature types...")
    feature_types = classify_features(esm2_ann, esm3_ann)
    print(f"  ESM-2: {feature_types['esm2']}")
    print(f"  ESM-3: {feature_types['esm3']}")

    # Identify convergent features with details
    print("\nIdentifying convergent features...")
    convergent = find_convergent_features(matches, esm2_ann, esm3_ann, threshold=0.95)
    print(f"  {len(convergent)} convergent feature pairs (sim >= 0.95)")

    if convergent:
        print("\n  Top 5 convergent features:")
        for cf in convergent[:5]:
            shared = ", ".join(cf["shared_enriched_aa"]) or "none"
            print(f"    ESM-2 #{cf['esm2_feature']} <-> ESM-3 #{cf['esm3_feature']}: "
                  f"sim={cf['similarity']:.4f}, shared AAs: {shared}")

    # Also compare at different similarity thresholds
    thresholds = [0.90, 0.92, 0.95, 0.97, 0.99]
    threshold_counts = {}
    for t in thresholds:
        n = sum(1 for m in matches if m["similarity"] >= t)
        threshold_counts[str(t)] = n
        print(f"  Features with sim >= {t}: {n}")

    # Characterize divergent features
    divergent = [m for m in matches if m["similarity"] < 0.90]
    divergent_esm2_types = defaultdict(int)
    divergent_esm3_types = defaultdict(int)

    for m in divergent:
        e2 = esm2_ann[m["esm2_idx"]]
        e3 = esm3_ann[m["esm3_idx"]]

        e2_enrich = [enr for _, enr in e2.get("top_enriched_aa", [])]
        e3_enrich = [enr for _, enr in e3.get("top_enriched_aa", [])]

        if max(e2_enrich, default=1.0) > 3.0:
            divergent_esm2_types["aa_specific"] += 1
        elif e2.get("functional_site_enrichment", 1.0) > 2.0:
            divergent_esm2_types["functional"] += 1
        else:
            divergent_esm2_types["generic"] += 1

        if max(e3_enrich, default=1.0) > 3.0:
            divergent_esm3_types["aa_specific"] += 1
        elif e3.get("functional_site_enrichment", 1.0) > 2.0:
            divergent_esm3_types["functional"] += 1
        else:
            divergent_esm3_types["generic"] += 1

    # Compile results
    results = {
        "comparison": "ESM-2 L24 vs ESM-3 S-only L33",
        "n_esm2_features": len(esm2_ann),
        "n_esm3_features": len(esm3_ann),
        "similarity_stats": {
            "mean": float(np.mean(match_sims)),
            "median": float(np.median(match_sims)),
            "std": float(np.std(match_sims)),
            "min": float(np.min(match_sims)),
            "max": float(np.max(match_sims)),
        },
        "convergent_counts": threshold_counts,
        "permutation_test": {
            "observed_mean": float(observed_mean),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "z_score": float(z_score),
            "p_value": float(p_val),
            "n_permutations": 1000,
        },
        "feature_types": feature_types,
        "divergent_feature_types": {
            "esm2": dict(divergent_esm2_types),
            "esm3": dict(divergent_esm3_types),
            "n_divergent": len(divergent),
        },
        "convergent_features": convergent,
        "top_matches": matches[:20],
    }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"ESM-2 (layer 24, 650M) vs ESM-3 (layer 33, 1.4B, seq-only)")
    print(f"Mean matched feature similarity: {observed_mean:.4f}")
    print(f"Convergent features (>0.95): {n_convergent_95}/{len(matches)} ({100*n_convergent_95/len(matches):.1f}%)")
    print(f"Permutation test: z={z_score:.1f}, p={p_val:.4f}")
    print(f"Feature type breakdown:")
    print(f"  ESM-2: {feature_types['esm2']}")
    print(f"  ESM-3: {feature_types['esm3']}")

    # Save results
    out_path = RESULTS_DIR / "esm2_esm3_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(sim_matrix, matches, esm2_ann, esm3_ann, feature_types, results)

    print("\nDone!")


if __name__ == "__main__":
    main()
