#!/usr/bin/env python3
"""Compute PCA of SAE decoder weights and merge per-feature metadata.

Produces decoder_pca.json with PCA coordinates and per-feature properties:
- GO enrichment (n_enriched_terms)
- Probe weight (functional site contribution)
- Cross-modal category (enhanced/suppressed/invariant)
- Convergence score (best-match Pearson r with ESM-2)

Usage:
    ./env/bin/python scripts/scaled_1.5M/39_decoder_pca.py
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
import torch
from sklearn.decomposition import PCA

RESULTS = ROOT / "results" / "scaled_1.5M"
SAE_PATH = ROOT / "models" / "sae_1.5M" / "esm3_residue_ef8_k64" / "best.pt"


def main():
    # 1. Load SAE decoder weights
    print("Loading SAE checkpoint...")
    ckpt = torch.load(SAE_PATH, map_location="cpu", weights_only=False)
    W_dec = ckpt["model_state_dict"]["decoder.weight"].float().numpy()  # (d_model=1536, d_sae=12288)
    d_model, d_sae = W_dec.shape
    print(f"Decoder: ({d_model}, {d_sae})")

    # 2. Find active features from residue-level SAE activation summaries
    print("Loading residue-level SAE summaries to identify active features...")
    with h5py.File(RESULTS / "sae_features" / "esm3" / "residue" / "protein_summaries.h5", "r") as f:
        feat_counts = f["feature_activation_counts"][:]  # (12288,) total residue activations
        esm3_acts = f["protein_max_activations"][:]  # (12491, 12288)
        esm3_pids = [x.decode() for x in f["protein_ids"][:]]

    active_mask = feat_counts > 0  # (12288,)
    active_indices = np.where(active_mask)[0]  # 0-based feature indices
    n_active = len(active_indices)
    print(f"Active features: {n_active} / {d_sae}")

    # 3. Compute PCA on active decoder columns
    W_active = W_dec[:, active_indices].T  # (n_active, d_model) — each row is a feature's decoder vector
    print(f"Computing PCA on {W_active.shape}...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(W_active)
    var_explained = pca.explained_variance_ratio_
    print(f"Variance explained: PC1={var_explained[0]:.4f}, PC2={var_explained[1]:.4f}, "
          f"total={sum(var_explained):.4f}")

    # 4. Compute convergence: best-match Pearson r with ESM-2
    print("Loading ESM-2 activations for convergence...")
    with h5py.File(RESULTS / "sae_features" / "esm2" / "residue" / "protein_summaries.h5", "r") as f:
        esm2_acts = f["protein_max_activations"][:]  # (12491, d_sae_esm2)
        esm2_pids = [x.decode() for x in f["protein_ids"][:]]

    # Align proteins
    esm3_pid_idx = {p: i for i, p in enumerate(esm3_pids)}
    esm2_pid_idx = {p: i for i, p in enumerate(esm2_pids)}
    shared = sorted(set(esm3_pids) & set(esm2_pids))
    print(f"Shared proteins: {len(shared)}")

    esm3_aligned = esm3_acts[[esm3_pid_idx[p] for p in shared]][:, active_indices]  # (n_shared, n_active)
    esm2_active = np.where((esm2_acts > 0).any(axis=0))[0]
    esm2_aligned = esm2_acts[[esm2_pid_idx[p] for p in shared]][:, esm2_active]  # (n_shared, n_esm2_active)

    print(f"Computing best-match correlations: {n_active} ESM-3 x {len(esm2_active)} ESM-2...")
    # Chunked computation to avoid memory issues
    chunk_size = 500
    best_r = np.zeros(n_active)
    for start in range(0, n_active, chunk_size):
        end = min(start + chunk_size, n_active)
        chunk = esm3_aligned[:, start:end]  # (n_shared, chunk)
        # Normalize
        chunk_mean = chunk.mean(axis=0, keepdims=True)
        chunk_std = chunk.std(axis=0, keepdims=True) + 1e-10
        chunk_norm = (chunk - chunk_mean) / chunk_std

        esm2_mean = esm2_aligned.mean(axis=0, keepdims=True)
        esm2_std = esm2_aligned.std(axis=0, keepdims=True) + 1e-10
        esm2_norm = (esm2_aligned - esm2_mean) / esm2_std

        corr = (chunk_norm.T @ esm2_norm) / len(shared)  # (chunk, n_esm2_active)
        best_r[start:end] = corr.max(axis=1)
        if (end % 2000 == 0) or end == n_active:
            print(f"  {end}/{n_active}")

    print(f"Convergence: median={np.median(best_r):.3f}, mean={np.mean(best_r):.3f}")

    # 5. Load per-feature metadata
    # Cross-modal categories
    cm = json.load(open(RESULTS / "cross_modal_features.json"))
    enhanced_ids = set(cm.get("enhanced_feature_ids", []))
    suppressed_ids = set(cm.get("suppressed_feature_ids", []))

    # GO enrichment
    go = json.load(open(RESULTS / "go_enrichment.json"))
    go_map = {f["feature_id"]: f.get("n_enriched_terms", 0)
              for f in go["esm3"].get("per_feature_results", [])}

    # Probe weights
    dg = json.load(open(RESULTS / "decoder_geometry.json"))
    probe_map = {m["feature_id"]: m.get("probe_weight", 0.0)
                 for m in dg["esm3"]["feature_metadata"]}

    # Activation counts from SAE activations
    act_counts = (esm3_acts[:, active_indices] > 0).sum(axis=0)  # proteins where feature fires

    # 6. Build merged metadata
    merged = []
    for i, feat_idx in enumerate(active_indices):
        fid = int(feat_idx) + 1  # 1-indexed feature ID

        if fid in enhanced_ids:
            category = "enhanced"
        elif fid in suppressed_ids:
            category = "suppressed"
        else:
            category = "invariant"

        merged.append({
            "feature_id": fid,
            "pca_x": float(coords[i, 0]),
            "pca_y": float(coords[i, 1]),
            "activation_count": int(act_counts[i]),
            "probe_weight": probe_map.get(fid, 0.0),
            "n_go_terms": go_map.get(fid, 0),
            "cross_modal_category": category,
            "convergence_r": float(best_r[i]),
        })

    output = {
        "model": "esm3",
        "sae_tag": "esm3_residue_ef8_k64",
        "method": "PCA",
        "n_components": 2,
        "variance_explained": [float(v) for v in var_explained],
        "total_variance_explained": float(sum(var_explained)),
        "d_sae": d_sae,
        "d_model": d_model,
        "n_active": len(merged),
        "feature_metadata": merged,
    }

    out_path = RESULTS / "decoder_pca.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {len(merged)} features to {out_path}")

    # Summary
    cats = {"enhanced": 0, "suppressed": 0, "invariant": 0}
    for m in merged:
        cats[m["cross_modal_category"]] += 1
    print(f"Categories: {cats}")
    go_arr = [m["n_go_terms"] for m in merged if m["n_go_terms"] > 0]
    print(f"GO enriched: {len(go_arr)}/{len(merged)}")


if __name__ == "__main__":
    main()
