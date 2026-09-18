#!/usr/bin/env python3
"""Compare SAE features vs PCA/NMF baselines on the same activations.

Shows that SAE produces more monosemantic, interpretable features than
dense decomposition methods.

Usage:
    ./env/bin/python scripts/unified/run_baseline_comparison.py --model esm2
    ./env/bin/python scripts/unified/run_baseline_comparison.py --model esm3
"""

import sys
import os
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
from sklearn.decomposition import PCA, NMF
from collections import Counter

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baseline_cmp")


def load_sae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def load_activations_subset(h5_path, n_residues=100000, seed=42):
    """Load a random subset of activations."""
    with h5py.File(h5_path, "r") as f:
        total = f["activations"].shape[0]
        rng = np.random.RandomState(seed)
        idx = rng.choice(total, min(n_residues, total), replace=False)
        idx.sort()
        acts = f["activations"][idx]
    return acts.astype(np.float32), idx


def build_residue_map(sequences, idx_subset):
    """Map global residue indices to (accession, position)."""
    accs = sorted(sequences.keys())
    offsets = []
    cur = 0
    for acc in accs:
        L = len(sequences[acc])
        offsets.append((acc, cur, cur + L))
        cur += L

    result = []
    oi = 0
    for global_idx in idx_subset:
        while oi < len(offsets) - 1 and global_idx >= offsets[oi + 1][1]:
            oi += 1
        acc, start, end = offsets[oi]
        pos = global_idx - start
        if pos < len(sequences[acc]):
            result.append((acc, pos))
        else:
            result.append((acc, 0))
    return result


def compute_interpretability(Z, residue_map, sequences, dssp_data, metadata,
                              method_name, n_components=None):
    """Compute interpretability metrics for a decomposition matrix Z (N, K)."""
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}
    N, K = Z.shape
    if n_components is not None:
        K = min(K, n_components)
        Z = Z[:, :K]

    # Build ground truth vectors
    aa_vec = np.zeros(N, dtype=int)  # AA index
    ss_vec = np.full(N, 2, dtype=int)  # 0=H, 1=E, 2=C
    func_vec = np.zeros(N, dtype=bool)

    # Build functional site sets
    func_sites = {}
    for acc in set(rm[0] for rm in residue_map):
        m = metadata.get(acc, {})
        sites = set()
        for feat in m.get("features", []):
            if feat.get("type") in ("Active site", "Binding site", "Disulfide bond"):
                for p in range(feat["start"] - 1, feat.get("end", feat["start"])):
                    sites.add(p)
        if sites:
            func_sites[acc] = sites

    for i, (acc, pos) in enumerate(residue_map):
        seq = sequences.get(acc, "")
        if pos < len(seq) and seq[pos] in aa_to_idx:
            aa_vec[i] = aa_to_idx[seq[pos]]
        dssp = dssp_data.get(acc, [])
        if dssp and pos < len(dssp):
            ss3 = dssp[pos].get("ss3", "C")
            ss_vec[i] = {"H": 0, "E": 1}.get(ss3, 2)
        if acc in func_sites and pos in func_sites[acc]:
            func_vec[i] = True

    # Background frequencies
    aa_bg = np.bincount(aa_vec, minlength=20).astype(float)
    aa_bg /= aa_bg.sum() + 1e-10
    ss_bg = np.bincount(ss_vec, minlength=3).astype(float)
    ss_bg /= ss_bg.sum() + 1e-10
    func_bg = func_vec.mean()

    # Per-feature metrics
    n_aa_specific = 0
    n_ss_specific = 0
    n_functional = 0
    kurtosis_vals = []
    gini_vals = []

    for j in range(K):
        col = Z[:, j]
        if col.max() <= 0:
            continue

        # Top 5% activated residues
        threshold = np.percentile(col[col > 0], 95) if (col > 0).sum() > 20 else col.max() * 0.5
        top_mask = col >= threshold
        n_top = top_mask.sum()
        if n_top < 5:
            continue

        # AA enrichment
        aa_top = np.bincount(aa_vec[top_mask], minlength=20).astype(float)
        aa_top /= aa_top.sum() + 1e-10
        aa_enr = aa_top / (aa_bg + 1e-10)
        if aa_enr.max() > 4.0:
            n_aa_specific += 1

        # SS enrichment
        ss_top = np.bincount(ss_vec[top_mask], minlength=3).astype(float)
        ss_top /= ss_top.sum() + 1e-10
        ss_enr = ss_top / (ss_bg + 1e-10)
        if ss_enr.max() > 1.5:
            n_ss_specific += 1

        # Functional enrichment
        func_top = func_vec[top_mask].mean()
        if func_bg > 0 and func_top / (func_bg + 1e-10) > 2.0 and func_top > 0.05:
            n_functional += 1

        # Monosemanticity (kurtosis of activations)
        active = col[col > 0]
        if len(active) > 10:
            from scipy.stats import kurtosis
            k_val = float(kurtosis(active, fisher=True))
            kurtosis_vals.append(k_val)

        # Gini coefficient
        if len(active) > 1:
            sorted_a = np.sort(active)
            n = len(sorted_a)
            index = np.arange(1, n + 1)
            gini = float((2 * (index * sorted_a).sum() / (n * sorted_a.sum())) - (n + 1) / n)
            gini_vals.append(gini)

    n_alive = sum(1 for j in range(K) if Z[:, j].max() > 0)

    return {
        "method": method_name,
        "n_components": K,
        "n_alive": n_alive,
        "n_aa_specific": n_aa_specific,
        "pct_aa_specific": 100 * n_aa_specific / max(n_alive, 1),
        "n_ss_specific": n_ss_specific,
        "pct_ss_specific": 100 * n_ss_specific / max(n_alive, 1),
        "n_functional": n_functional,
        "pct_functional": 100 * n_functional / max(n_alive, 1),
        "mean_kurtosis": float(np.mean(kurtosis_vals)) if kurtosis_vals else 0,
        "median_kurtosis": float(np.median(kurtosis_vals)) if kurtosis_vals else 0,
        "mean_gini": float(np.mean(gini_vals)) if gini_vals else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], default="esm2")
    parser.add_argument("--n-residues", type=int, default=100000)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "unified" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json") as f:
        dssp_data = json.load(f)

    if args.model == "esm2":
        h5_path = ROOT / "data" / "activations" / "esm2_scaled" / "layer_24.h5"
        sae_path = ROOT / "models" / "sae" / "esm2_scaled" / "layer_24_topk" / "best.pt"
        layer = 24
        d_model = 1280
    else:
        h5_path = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S_layer_33.h5"
        sae_path = ROOT / "models" / "sae" / "esm3" / "S_layer_33_topk" / "best.pt"
        layer = 33
        d_model = 1536

    # Load activations
    log.info(f"Loading {args.n_residues} residues from {h5_path}...")
    X, idx = load_activations_subset(str(h5_path), n_residues=args.n_residues)
    residue_map = build_residue_map(sequences, idx)
    log.info(f"Loaded: {X.shape}")

    results = {"model": args.model, "layer": layer, "n_residues": len(X), "methods": []}

    # ═══ SAE ═══
    log.info("Encoding through SAE...")
    sae = load_sae(str(sae_path))
    with torch.no_grad():
        Z_sae = sae.encode(torch.tensor(X)).numpy()
    log.info(f"SAE features: {Z_sae.shape}")

    sae_result = compute_interpretability(
        Z_sae, residue_map, sequences, dssp_data, metadata, "SAE (TopK)")
    results["methods"].append(sae_result)
    log.info(f"SAE: {sae_result['pct_aa_specific']:.1f}% AA-specific, "
             f"{sae_result['pct_ss_specific']:.1f}% SS-specific, "
             f"{sae_result['pct_functional']:.1f}% functional, "
             f"kurtosis={sae_result['mean_kurtosis']:.1f}")

    # ═══ PCA ═══
    n_pca = min(d_model, len(X) - 1)
    log.info(f"Fitting PCA with {n_pca} components...")
    pca = PCA(n_components=n_pca, random_state=42)
    Z_pca = pca.fit_transform(X)
    # Take absolute values (PCA components can be negative)
    Z_pca_abs = np.abs(Z_pca)
    log.info(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")

    pca_result = compute_interpretability(
        Z_pca_abs, residue_map, sequences, dssp_data, metadata, "PCA",
        n_components=n_pca)
    pca_result["variance_explained"] = float(pca.explained_variance_ratio_.sum())
    results["methods"].append(pca_result)
    log.info(f"PCA: {pca_result['pct_aa_specific']:.1f}% AA-specific, "
             f"{pca_result['pct_ss_specific']:.1f}% SS-specific, "
             f"{pca_result['pct_functional']:.1f}% functional, "
             f"kurtosis={pca_result['mean_kurtosis']:.1f}")

    # ═══ NMF ═══
    n_nmf = min(d_model, len(X) - 1)
    log.info(f"Fitting NMF with {n_nmf} components...")
    # NMF requires non-negative input: shift by column min
    X_shift = X - X.min(axis=0, keepdims=True)
    try:
        nmf = NMF(n_components=n_nmf, init="nndsvda", max_iter=300, random_state=42)
        Z_nmf = nmf.fit_transform(X_shift)
        recon_err = float(nmf.reconstruction_err_)
        log.info(f"NMF reconstruction error: {recon_err:.2f}")

        nmf_result = compute_interpretability(
            Z_nmf, residue_map, sequences, dssp_data, metadata, "NMF",
            n_components=n_nmf)
        nmf_result["reconstruction_error"] = recon_err
        results["methods"].append(nmf_result)
        log.info(f"NMF: {nmf_result['pct_aa_specific']:.1f}% AA-specific, "
                 f"{nmf_result['pct_ss_specific']:.1f}% SS-specific, "
                 f"{nmf_result['pct_functional']:.1f}% functional, "
                 f"kurtosis={nmf_result['mean_kurtosis']:.1f}")
    except Exception as e:
        log.error(f"NMF failed: {e}")
        results["methods"].append({"method": "NMF", "error": str(e)})

    # ═══ Summary ═══
    log.info("\n" + "=" * 60)
    log.info(f"{'Method':<15} {'N':>6} {'AA%':>6} {'SS%':>6} {'Func%':>6} {'Kurt':>7} {'Gini':>6}")
    log.info("-" * 60)
    for m in results["methods"]:
        if "error" in m:
            log.info(f"{m['method']:<15} ERROR: {m['error'][:40]}")
            continue
        log.info(f"{m['method']:<15} {m['n_alive']:>6} {m['pct_aa_specific']:>5.1f}% "
                 f"{m['pct_ss_specific']:>5.1f}% {m['pct_functional']:>5.1f}% "
                 f"{m['mean_kurtosis']:>7.1f} {m['mean_gini']:>5.2f}")

    out_path = out_dir / f"baseline_comparison_L{layer}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
