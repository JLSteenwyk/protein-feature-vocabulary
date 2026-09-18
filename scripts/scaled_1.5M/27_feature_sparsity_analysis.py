#!/usr/bin/env python3
"""Feature sparsity/density analysis for SAE features.

Standard SAE analysis: distribution of features-per-token (L0),
tokens-per-feature (activation frequency), and activation magnitudes.

Usage:
    ./env/bin/python scripts/scaled_1.5M/27_feature_sparsity_analysis.py
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
from scipy import sparse
import h5py

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sparsity")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"


def analyze_model(model_name, d_sae):
    log.info(f"\n{'='*60}")
    log.info(f"  {model_name.upper()} SPARSITY ANALYSIS")
    log.info(f"{'='*60}")

    results = {}

    # --- Residue-level analysis ---
    sparse_path = FEATURE_ROOT / model_name / "residue" / "features_sparse.npz"
    summary_path = FEATURE_ROOT / model_name / "residue" / "protein_summaries.h5"

    if sparse_path.exists():
        log.info("Loading residue-level sparse features...")
        X = sparse.load_npz(sparse_path)
        n_residues, n_features = X.shape
        log.info(f"  Shape: {n_residues:,} residues x {n_features:,} features")

        # 1. Features-per-token (L0) distribution
        log.info("Computing features-per-token (L0)...")
        # For CSR, number of nonzeros per row
        l0_per_residue = np.diff(X.indptr)  # nnz per row
        l0_stats = {
            "mean": float(np.mean(l0_per_residue)),
            "median": float(np.median(l0_per_residue)),
            "std": float(np.std(l0_per_residue)),
            "min": int(np.min(l0_per_residue)),
            "max": int(np.max(l0_per_residue)),
            "percentiles": {str(p): float(np.percentile(l0_per_residue, p))
                           for p in [5, 10, 25, 50, 75, 90, 95]},
            "histogram": {},
        }
        # Histogram of L0 values
        hist_vals, hist_edges = np.histogram(l0_per_residue, bins=50)
        l0_stats["histogram"]["counts"] = hist_vals.tolist()
        l0_stats["histogram"]["bin_edges"] = hist_edges.tolist()
        log.info(f"  L0: mean={l0_stats['mean']:.1f}, median={l0_stats['median']:.0f}, "
                 f"min={l0_stats['min']}, max={l0_stats['max']}")

        # 2. Tokens-per-feature (activation frequency)
        log.info("Computing tokens-per-feature...")
        # For CSC, count nnz per column
        X_csc = X.tocsc()
        nnz_per_feature = np.diff(X_csc.indptr)  # nnz per column
        active_mask = nnz_per_feature > 0
        n_active = active_mask.sum()
        n_dead = n_features - n_active

        freq_stats = {
            "n_active_features": int(n_active),
            "n_dead_features": int(n_dead),
            "dead_fraction": float(n_dead / n_features),
            "active_features": {},
        }

        # Stats only for active features
        active_nnz = nnz_per_feature[active_mask]
        freq_stats["active_features"] = {
            "mean_tokens": float(np.mean(active_nnz)),
            "median_tokens": float(np.median(active_nnz)),
            "std_tokens": float(np.std(active_nnz)),
            "min_tokens": int(np.min(active_nnz)),
            "max_tokens": int(np.max(active_nnz)),
            "mean_frac": float(np.mean(active_nnz) / n_residues),
            "percentiles": {str(p): float(np.percentile(active_nnz, p))
                           for p in [5, 10, 25, 50, 75, 90, 95, 99]},
        }
        # Histogram (log-scale bins)
        log_bins = np.logspace(0, np.log10(max(active_nnz) + 1), 50)
        hist_vals, hist_edges = np.histogram(active_nnz, bins=log_bins)
        freq_stats["active_features"]["histogram"] = {
            "counts": hist_vals.tolist(),
            "bin_edges": hist_edges.tolist(),
        }
        log.info(f"  Active: {n_active:,}/{n_features:,} ({100*n_active/n_features:.1f}%)")
        log.info(f"  Tokens/feature: mean={np.mean(active_nnz):.0f}, "
                 f"median={np.median(active_nnz):.0f}")

        # 3. Activation magnitude distribution
        log.info("Computing activation magnitudes...")
        all_vals = X.data  # all nonzero values
        mag_stats = {
            "mean": float(np.mean(all_vals)),
            "median": float(np.median(all_vals)),
            "std": float(np.std(all_vals)),
            "min": float(np.min(all_vals)),
            "max": float(np.max(all_vals)),
            "percentiles": {str(p): float(np.percentile(all_vals, p))
                           for p in [5, 10, 25, 50, 75, 90, 95, 99]},
        }
        hist_vals, hist_edges = np.histogram(all_vals, bins=100,
                                              range=(0, np.percentile(all_vals, 99)))
        mag_stats["histogram"] = {
            "counts": hist_vals.tolist(),
            "bin_edges": hist_edges.tolist(),
        }
        log.info(f"  Magnitudes: mean={mag_stats['mean']:.3f}, "
                 f"median={mag_stats['median']:.3f}, max={mag_stats['max']:.3f}")

        # 4. Overall sparsity
        total_elements = n_residues * n_features
        nnz_total = X.nnz
        sparsity = {
            "n_residues": int(n_residues),
            "n_features": int(n_features),
            "total_elements": int(total_elements),
            "nnz": int(nnz_total),
            "density": float(nnz_total / total_elements),
            "sparsity": float(1 - nnz_total / total_elements),
        }
        log.info(f"  Sparsity: {100*sparsity['sparsity']:.2f}% zeros, "
                 f"density={100*sparsity['density']:.4f}%")

        results["residue"] = {
            "l0_distribution": l0_stats,
            "feature_frequency": freq_stats,
            "activation_magnitude": mag_stats,
            "sparsity": sparsity,
        }

    # --- Protein-level analysis ---
    protein_path = FEATURE_ROOT / model_name / "protein" / "features.h5"
    if protein_path.exists():
        log.info("\nProtein-level features...")
        with h5py.File(protein_path, "r") as f:
            Z = f["features"][:]
        n_proteins, n_feat = Z.shape
        active_per_protein = (Z > 0).sum(axis=1)
        active_per_feature = (Z > 0).sum(axis=0)
        n_active_prot = (active_per_feature > 0).sum()

        results["protein"] = {
            "n_proteins": int(n_proteins),
            "n_features": int(n_feat),
            "n_active_features": int(n_active_prot),
            "features_per_protein": {
                "mean": float(np.mean(active_per_protein)),
                "median": float(np.median(active_per_protein)),
                "std": float(np.std(active_per_protein)),
            },
            "proteins_per_feature": {
                "mean": float(np.mean(active_per_feature[active_per_feature > 0])),
                "median": float(np.median(active_per_feature[active_per_feature > 0])),
            },
        }
        log.info(f"  Active features: {n_active_prot}/{n_feat}")
        log.info(f"  Features/protein: mean={np.mean(active_per_protein):.1f}")

    return results


def main():
    results = {}
    for model_name, d_sae in [("esm3", 12288), ("esm2", 10240)]:
        results[model_name] = analyze_model(model_name, d_sae)

    out_path = OUTPUT_DIR / "feature_sparsity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
