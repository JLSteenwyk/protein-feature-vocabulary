#!/usr/bin/env python3
"""ESM-2 vs ESM-3 SAE feature convergence analysis.

Compares SAE features across models by correlating their activation patterns
on the shared eval protein set.

Usage:
    ./env/bin/python scripts/scaled_1.5M/12_feature_convergence.py
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
from scipy import stats

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("convergence")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"


def _chunked_best_match(source_mat, target_mat, chunk_size=500, return_idx=False):
    """Compute best-match Pearson r from source to target features.

    Args:
        source_mat: (n_proteins, n_source_features) — query features
        target_mat: (n_proteins, n_target_features) — search space
        chunk_size: number of source features per chunk
        return_idx: if True, also return indices of best matches

    Returns:
        best_r: array of shape (n_source_features,)
        best_idx: array of shape (n_source_features,) [only if return_idx=True]
    """
    n_proteins = source_mat.shape[0]
    n_source = source_mat.shape[1]

    best_r = np.zeros(n_source)
    best_idx = np.zeros(n_source, dtype=np.int64) if return_idx else None

    # Pre-normalize target (constant across chunks)
    target_norm = target_mat.T - target_mat.T.mean(axis=1, keepdims=True)
    target_std = target_norm.std(axis=1, keepdims=True)
    target_std[target_std == 0] = 1
    target_norm /= target_std

    for i in range(0, n_source, chunk_size):
        chunk_end = min(i + chunk_size, n_source)
        src_chunk = source_mat[:, i:chunk_end].T  # (chunk, n_proteins)

        src_norm = src_chunk - src_chunk.mean(axis=1, keepdims=True)
        src_std = src_norm.std(axis=1, keepdims=True)
        src_std[src_std == 0] = 1
        src_norm /= src_std

        corr = src_norm @ target_norm.T / n_proteins
        best_r[i:chunk_end] = corr.max(axis=1)
        if return_idx:
            best_idx[i:chunk_end] = corr.argmax(axis=1)

    if return_idx:
        return best_r, best_idx
    return best_r


def _rank_transform(mat):
    """Rank-transform each column (feature) of the matrix.

    Ties get average rank. Result has same shape as input.
    Uses scipy.stats.rankdata per column for correctness.
    """
    ranked = np.empty_like(mat, dtype=np.float64)
    for j in range(mat.shape[1]):
        ranked[:, j] = stats.rankdata(mat[:, j], method='average')
    return ranked


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load protein-level features for both models
    log.info("Loading protein-level features...")
    models = {}
    for model_name in ["esm3", "esm2"]:
        feat_path = FEATURE_ROOT / model_name / "residue" / "protein_summaries.h5"
        if not feat_path.exists():
            log.warning(f"No features for {model_name}")
            continue
        with h5py.File(feat_path, "r") as f:
            models[model_name] = {
                "ids": [s.decode() for s in f["protein_ids"][:]],
                "max_acts": f["protein_max_activations"][:],
                "mean_acts": f["protein_mean_activations"][:],
            }

    if len(models) < 2:
        log.error("Need both ESM-2 and ESM-3 features")
        return

    # Find shared proteins
    esm3_ids = models["esm3"]["ids"]
    esm2_ids = models["esm2"]["ids"]
    shared = set(esm3_ids) & set(esm2_ids)
    log.info(f"Shared proteins: {len(shared)} (ESM-3: {len(esm3_ids)}, ESM-2: {len(esm2_ids)})")

    if len(shared) < 100:
        log.error("Too few shared proteins")
        return

    # Build aligned matrices
    esm3_idx = {pid: i for i, pid in enumerate(esm3_ids)}
    esm2_idx = {pid: i for i, pid in enumerate(esm2_ids)}
    shared_list = sorted(shared)

    esm3_aligned = models["esm3"]["max_acts"][[esm3_idx[p] for p in shared_list]]
    esm2_aligned = models["esm2"]["max_acts"][[esm2_idx[p] for p in shared_list]]

    # Filter to active features only
    esm3_active = np.where((esm3_aligned > 0).any(axis=0))[0]
    esm2_active = np.where((esm2_aligned > 0).any(axis=0))[0]
    log.info(f"Active features: ESM-3={len(esm3_active)}, ESM-2={len(esm2_active)}")

    esm3_mat = esm3_aligned[:, esm3_active]  # (n_shared, n_esm3_active)
    esm2_mat = esm2_aligned[:, esm2_active]  # (n_shared, n_esm2_active)

    # Compute cross-correlation: for each ESM-3 feature, find best-matching ESM-2 feature
    log.info("Computing cross-model feature correlations (Pearson)...")
    log.info(f"  Matrix: {len(esm3_active)} ESM-3 x {len(esm2_active)} ESM-2 features")

    best_match_r, best_match_idx = _chunked_best_match(esm3_mat, esm2_mat, return_idx=True)
    log.info(f"  ESM-3 -> ESM-2 done. Median best r = {np.median(best_match_r):.3f}")

    best_match_r_esm2 = _chunked_best_match(esm2_mat, esm3_mat)
    log.info(f"  ESM-2 -> ESM-3 done. Median best r = {np.median(best_match_r_esm2):.3f}")

    # --- Spearman rank correlation (M5) ---
    log.info("Rank-transforming feature matrices for Spearman...")
    esm3_ranked = _rank_transform(esm3_mat)
    esm2_ranked = _rank_transform(esm2_mat)
    log.info("  Rank transform done. Computing Spearman best-match correlations...")

    spearman_best_r = _chunked_best_match(esm3_ranked, esm2_ranked)
    log.info(f"  ESM-3 -> ESM-2 Spearman done. Median best rho = {np.median(spearman_best_r):.3f}")

    spearman_best_r_esm2 = _chunked_best_match(esm2_ranked, esm3_ranked)
    log.info(f"  ESM-2 -> ESM-3 Spearman done. Median best rho = {np.median(spearman_best_r_esm2):.3f}")

    # --- Permutation null distribution (C3) ---
    n_permutations = 10
    log.info(f"Running permutation test ({n_permutations} permutations)...")
    rng = np.random.default_rng(42)
    null_best_r_all = []  # collect all per-feature best-r values across permutations

    for perm_i in range(n_permutations):
        perm_idx = rng.permutation(esm3_mat.shape[0])
        esm3_shuffled = esm3_mat[perm_idx, :]
        null_r = _chunked_best_match(esm3_shuffled, esm2_mat)
        null_best_r_all.append(null_r)
        log.info(f"  Permutation {perm_i+1}/{n_permutations}: "
                 f"median best r = {np.median(null_r):.3f}, "
                 f"frac >= 0.3: {(null_r >= 0.3).mean():.3f}")

    null_best_r_all = np.concatenate(null_best_r_all)
    null_distribution = {
        "n_permutations": n_permutations,
        "median_best_r": float(np.median(null_best_r_all)),
        "mean_best_r": float(np.mean(null_best_r_all)),
        "frac_above_thresholds": {
            str(t): float((null_best_r_all >= t).mean())
            for t in [0.3, 0.5, 0.7, 0.9]
        },
        "percentiles": {
            str(p): float(np.percentile(null_best_r_all, p))
            for p in [50, 75, 90, 95]
        },
    }
    log.info(f"  Null distribution: median={null_distribution['median_best_r']:.3f}, "
             f"frac>=0.3={null_distribution['frac_above_thresholds']['0.3']:.3f}")

    # Summary statistics
    thresholds = [0.3, 0.5, 0.7, 0.9]
    log.info(f"\n{'='*60}")
    log.info("FEATURE CONVERGENCE RESULTS")
    log.info(f"{'='*60}")

    log.info(f"\nESM-3 → ESM-2 best-match correlations (Pearson):")
    log.info(f"  Median r: {np.median(best_match_r):.3f}")
    log.info(f"  Mean r: {np.mean(best_match_r):.3f}")
    for t in thresholds:
        frac = (best_match_r >= t).mean()
        log.info(f"  r >= {t}: {(best_match_r >= t).sum()} ({100*frac:.1f}%)")

    log.info(f"\nESM-3 → ESM-2 best-match correlations (Spearman):")
    log.info(f"  Median rho: {np.median(spearman_best_r):.3f}")
    log.info(f"  Mean rho: {np.mean(spearman_best_r):.3f}")
    for t in thresholds:
        frac = (spearman_best_r >= t).mean()
        log.info(f"  rho >= {t}: {(spearman_best_r >= t).sum()} ({100*frac:.1f}%)")

    log.info(f"\nESM-2 → ESM-3 best-match correlations (Pearson):")
    log.info(f"  Median r: {np.median(best_match_r_esm2):.3f}")
    log.info(f"  Mean r: {np.mean(best_match_r_esm2):.3f}")
    for t in thresholds:
        frac = (best_match_r_esm2 >= t).mean()
        log.info(f"  r >= {t}: {(best_match_r_esm2 >= t).sum()} ({100*frac:.1f}%)")

    log.info(f"\nESM-2 → ESM-3 best-match correlations (Spearman):")
    log.info(f"  Median rho: {np.median(spearman_best_r_esm2):.3f}")
    log.info(f"  Mean rho: {np.mean(spearman_best_r_esm2):.3f}")
    for t in thresholds:
        frac = (spearman_best_r_esm2 >= t).mean()
        log.info(f"  rho >= {t}: {(spearman_best_r_esm2 >= t).sum()} ({100*frac:.1f}%)")

    log.info(f"\nPermutation null (ESM-3 → ESM-2, {n_permutations} perms):")
    log.info(f"  Median best r: {null_distribution['median_best_r']:.3f}")
    log.info(f"  Mean best r: {null_distribution['mean_best_r']:.3f}")
    for t in thresholds:
        frac = null_distribution["frac_above_thresholds"].get(str(t), 0)
        log.info(f"  r >= {t}: {100*frac:.1f}%")

    output = {
        "n_shared_proteins": len(shared_list),
        "n_esm3_active": len(esm3_active),
        "n_esm2_active": len(esm2_active),
        "esm3_to_esm2": {
            "pearson": {
                "median_best_r": float(np.median(best_match_r)),
                "mean_best_r": float(np.mean(best_match_r)),
                "frac_above_thresholds": {
                    str(t): float((best_match_r >= t).mean()) for t in thresholds
                },
                "r_distribution": {
                    "percentiles": {str(p): float(np.percentile(best_match_r, p))
                                    for p in [10, 25, 50, 75, 90]},
                    "values": [float(r) for r in best_match_r],
                },
            },
            "spearman": {
                "median_best_r": float(np.median(spearman_best_r)),
                "mean_best_r": float(np.mean(spearman_best_r)),
                "frac_above_thresholds": {
                    str(t): float((spearman_best_r >= t).mean()) for t in thresholds
                },
                "r_distribution": {
                    "percentiles": {str(p): float(np.percentile(spearman_best_r, p))
                                    for p in [10, 25, 50, 75, 90]},
                    "values": [float(r) for r in spearman_best_r],
                },
            },
            # Legacy top-level keys for backward compatibility
            "median_best_r": float(np.median(best_match_r)),
            "mean_best_r": float(np.mean(best_match_r)),
            "frac_above_thresholds": {
                str(t): float((best_match_r >= t).mean()) for t in thresholds
            },
        },
        "esm2_to_esm3": {
            "pearson": {
                "median_best_r": float(np.median(best_match_r_esm2)),
                "mean_best_r": float(np.mean(best_match_r_esm2)),
                "frac_above_thresholds": {
                    str(t): float((best_match_r_esm2 >= t).mean()) for t in thresholds
                },
                "r_distribution": {
                    "percentiles": {str(p): float(np.percentile(best_match_r_esm2, p))
                                    for p in [10, 25, 50, 75, 90]},
                    "values": [float(r) for r in best_match_r_esm2],
                },
            },
            "spearman": {
                "median_best_r": float(np.median(spearman_best_r_esm2)),
                "mean_best_r": float(np.mean(spearman_best_r_esm2)),
                "frac_above_thresholds": {
                    str(t): float((spearman_best_r_esm2 >= t).mean()) for t in thresholds
                },
                "r_distribution": {
                    "percentiles": {str(p): float(np.percentile(spearman_best_r_esm2, p))
                                    for p in [10, 25, 50, 75, 90]},
                    "values": [float(r) for r in spearman_best_r_esm2],
                },
            },
            # Legacy top-level keys for backward compatibility
            "median_best_r": float(np.median(best_match_r_esm2)),
            "mean_best_r": float(np.mean(best_match_r_esm2)),
            "frac_above_thresholds": {
                str(t): float((best_match_r_esm2 >= t).mean()) for t in thresholds
            },
        },
        "null_distribution": null_distribution,
    }

    out_path = OUTPUT_DIR / "feature_convergence.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
