#!/usr/bin/env python3
"""Co-activation analysis: identify modules of SAE features that fire together.

Usage:
    ./env/bin/python scripts/scaled_1.5M/15_coactivation_analysis.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
from scipy import sparse
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("coactivation")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MAX_RESIDUES = 200_000  # subsample for correlation computation


def compute_jaccard_from_sparse(X_sparse, active_features, max_residues=MAX_RESIDUES):
    """Compute pairwise Jaccard similarity of feature activation patterns."""
    rng = np.random.RandomState(42)

    # Subsample residues
    if X_sparse.shape[0] > max_residues:
        idx = rng.choice(X_sparse.shape[0], max_residues, replace=False)
        X = X_sparse[idx]
    else:
        X = X_sparse

    # Filter to active features
    X = X[:, active_features]
    log.info(f"  Computing Jaccard on {X.shape[0]} residues x {X.shape[1]} features")

    # Binarize
    X_bin = (X > 0).astype(np.float32)

    # Convert to CSC for efficient column operations
    X_csc = sparse.csc_matrix(X_bin)

    # Co-activation: X.T @ X gives co-occurrence counts
    cooccur = (X_csc.T @ X_csc).toarray().astype(np.float64)

    # Column sums = activation counts per feature
    col_sums = np.array(X_csc.sum(axis=0)).flatten().astype(np.float64)

    # Jaccard: |A ∩ B| / |A ∪ B| = cooccur / (sum_A + sum_B - cooccur)
    n = len(active_features)
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            union = col_sums[i] + col_sums[j] - cooccur[i, j]
            if union > 0:
                jaccard[i, j] = cooccur[i, j] / union
                jaccard[j, i] = jaccard[i, j]

    return jaccard


def run_coactivation(model_name):
    """Run co-activation analysis for one model."""
    log.info(f"\n{'='*60}")
    log.info(f"CO-ACTIVATION ANALYSIS: {model_name}")
    log.info(f"{'='*60}")

    # Load sparse features
    feat_dir = FEATURE_ROOT / model_name / "residue"
    X_sparse = sparse.load_npz(str(feat_dir / "features_sparse.npz"))

    with h5py.File(feat_dir / "protein_summaries.h5", "r") as f:
        feature_counts = f["feature_activation_counts"][:]

    active_features = np.where(feature_counts > 0)[0]
    log.info(f"  {X_sparse.shape[0]} residues, {len(active_features)} active features")

    # Cap active features for tractability
    if len(active_features) > 2000:
        # Take top 2000 by activation frequency
        top_idx = np.argsort(-feature_counts[active_features])[:2000]
        active_features = active_features[top_idx]
        log.info(f"  Capped to top 2000 features by frequency")

    # Compute Jaccard similarity
    jaccard = compute_jaccard_from_sparse(X_sparse, active_features)
    log.info(f"  Jaccard matrix: {jaccard.shape}")

    # Hierarchical clustering
    dist = 1 - jaccard
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, 1)

    # Convert to condensed form
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")

    # Cut at different granularities
    results = {}
    for n_clusters in [10, 20, 30]:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(int(active_features[i]))

        # Cluster statistics
        cluster_stats = []
        for cid, features in sorted(clusters.items()):
            # Mean within-cluster Jaccard
            if len(features) > 1:
                feat_idx = [np.where(active_features == f)[0][0] for f in features]
                sub_jaccard = jaccard[np.ix_(feat_idx, feat_idx)]
                mean_jaccard = sub_jaccard[np.triu_indices(len(features), k=1)].mean()
            else:
                mean_jaccard = 0.0

            cluster_stats.append({
                "cluster_id": cid,
                "n_features": len(features),
                "feature_ids": features[:20],  # cap for JSON size
                "mean_within_jaccard": float(mean_jaccard),
            })

        results[f"k{n_clusters}"] = {
            "n_clusters": n_clusters,
            "clusters": cluster_stats,
        }
        log.info(f"  k={n_clusters}: cluster sizes = "
                 f"{sorted([s['n_features'] for s in cluster_stats], reverse=True)[:5]}...")

    # Overall statistics
    upper_tri = jaccard[np.triu_indices(len(active_features), k=1)]
    log.info(f"  Mean pairwise Jaccard: {upper_tri.mean():.4f}")
    log.info(f"  Jaccard > 0.1: {(upper_tri > 0.1).sum()} pairs")
    log.info(f"  Jaccard > 0.3: {(upper_tri > 0.3).sum()} pairs")

    return {
        "model": model_name,
        "n_active_features": len(active_features),
        "mean_pairwise_jaccard": float(upper_tri.mean()),
        "n_pairs_above_0.1": int((upper_tri > 0.1).sum()),
        "n_pairs_above_0.3": int((upper_tri > 0.3).sum()),
        "clustering": results,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name in ["esm3", "esm2"]:
        feat_path = FEATURE_ROOT / model_name / "residue" / "features_sparse.npz"
        if not feat_path.exists():
            log.warning(f"No features for {model_name}")
            continue
        results[model_name] = run_coactivation(model_name)

    out_path = OUTPUT_DIR / "coactivation_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
