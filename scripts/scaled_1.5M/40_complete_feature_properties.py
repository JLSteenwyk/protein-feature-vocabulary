#!/usr/bin/env python3
"""Build complete per-feature property dataset for all active features.

Re-runs cross-modal classification to get ALL feature IDs (not just top 200),
then merges with convergence scores, GO enrichment, and probe weights.

Usage:
    ./env/bin/python scripts/scaled_1.5M/40_complete_feature_properties.py
"""
import os, sys, json
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
import torch
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feature_props")

RESULTS = ROOT / "results" / "scaled_1.5M"
CM_DIR = RESULTS / "cross_modal"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
SAE_TAG = "esm3_residue_ef8_k64"


def classify_cross_modal():
    """Re-run cross-modal classification, returning per-feature arrays."""
    from sae.model import build_sae

    sae_path = MODEL_ROOT / SAE_TAG / "best.pt"
    ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    d_sae = ckpt["sae_config"].input_dim * ckpt["sae_config"].expansion_factor
    log.info(f"SAE loaded: d_sae={d_sae}")

    s_dir = CM_DIR / "S_only" / "residue_L33"
    sst_dir = CM_DIR / "S_St" / "residue_L33"
    s_chunks = sorted(s_dir.glob("*.h5"))
    sst_chunks = sorted(sst_dir.glob("*.h5"))

    protein_mean_S = []
    protein_mean_SSt = []

    for s_cp, sst_cp in zip(s_chunks, sst_chunks):
        with h5py.File(s_cp, "r") as f:
            s_acts = torch.tensor(f["activations"][:].astype(np.float32))
            s_offsets = f["offsets"][:]
            ids = [s.decode() for s in f["ids"][:]]

        with h5py.File(sst_cp, "r") as f:
            sst_acts = torch.tensor(f["activations"][:].astype(np.float32))
            sst_offsets = f["offsets"][:]

        batch_size = 4096
        z_s_list = []
        for i in range(0, s_acts.shape[0], batch_size):
            with torch.no_grad():
                z_s_list.append(sae.encode(s_acts[i:i+batch_size]).numpy())
        Z_s = np.concatenate(z_s_list, axis=0)

        z_sst_list = []
        for i in range(0, sst_acts.shape[0], batch_size):
            with torch.no_grad():
                z_sst_list.append(sae.encode(sst_acts[i:i+batch_size]).numpy())
        Z_sst = np.concatenate(z_sst_list, axis=0)

        for pi in range(len(ids)):
            _, off_s, L_s = s_offsets[pi]
            _, off_sst, L_sst = sst_offsets[pi]
            L = min(L_s, L_sst)
            protein_mean_S.append(Z_s[off_s:off_s+L].mean(axis=0))
            protein_mean_SSt.append(Z_sst[off_sst:off_sst+L].mean(axis=0))

    protein_mean_S = np.stack(protein_mean_S)
    protein_mean_SSt = np.stack(protein_mean_SSt)
    n_proteins = protein_mean_S.shape[0]
    log.info(f"Encoded {n_proteins} proteins")

    delta = protein_mean_SSt - protein_mean_S
    active_either = (protein_mean_S > 0).any(axis=0) | (protein_mean_SSt > 0).any(axis=0)
    active_features = np.where(active_either)[0]
    log.info(f"Active features (either condition): {len(active_features)}")

    p_values = np.ones(d_sae)
    effect_sizes = np.zeros(d_sae)
    mean_deltas = np.zeros(d_sae)

    for fid in active_features:
        d = delta[:, fid]
        nonzero = d[d != 0]
        if len(nonzero) < 10:
            continue
        try:
            _, p = wilcoxon(d[d != 0])
            p_values[fid] = p
        except Exception:
            pass
        mean_d = d.mean()
        std_d = d.std()
        if std_d > 0:
            effect_sizes[fid] = mean_d / std_d
        mean_deltas[fid] = mean_d

    tested_idx = np.where(p_values < 1.0)[0]
    p_corrected = np.ones(d_sae)
    if len(tested_idx) > 0:
        _, corrected_p, _, _ = multipletests(p_values[tested_idx], method="fdr_bh")
        p_corrected[tested_idx] = corrected_p

    sig_mask = p_corrected < 0.05
    enhanced = sig_mask & (effect_sizes > 0.2)
    suppressed = sig_mask & (effect_sizes < -0.2)
    invariant = ~enhanced & ~suppressed & active_either

    log.info(f"Enhanced: {enhanced.sum()}, Suppressed: {suppressed.sum()}, "
             f"Invariant: {invariant.sum()}")

    # Build per-feature category array (0=inactive, 1=enhanced, 2=suppressed, 3=invariant)
    categories = np.zeros(d_sae, dtype=int)
    categories[enhanced] = 1
    categories[suppressed] = 2
    categories[invariant] = 3

    return categories, effect_sizes, mean_deltas, p_corrected, d_sae


def main():
    # 1. Cross-modal classification (ALL features)
    log.info("=== Cross-modal classification ===")
    categories, effect_sizes, mean_deltas, p_corrected, d_sae = classify_cross_modal()

    # 2. Convergence scores
    log.info("=== Loading convergence data ===")
    with h5py.File(RESULTS / "sae_features" / "esm3" / "residue" / "protein_summaries.h5", "r") as f:
        feat_counts = f["feature_activation_counts"][:]
        esm3_max = f["protein_max_activations"][:]
        esm3_pids = [x.decode() for x in f["protein_ids"][:]]

    with h5py.File(RESULTS / "sae_features" / "esm2" / "residue" / "protein_summaries.h5", "r") as f:
        esm2_max = f["protein_max_activations"][:]
        esm2_pids = [x.decode() for x in f["protein_ids"][:]]

    active_mask = feat_counts > 0
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)
    log.info(f"Active features: {n_active}")

    # Align proteins
    esm3_idx = {p: i for i, p in enumerate(esm3_pids)}
    esm2_idx = {p: i for i, p in enumerate(esm2_pids)}
    shared = sorted(set(esm3_pids) & set(esm2_pids))

    esm3_aligned = esm3_max[[esm3_idx[p] for p in shared]][:, active_indices]
    esm2_active = np.where((esm2_max > 0).any(axis=0))[0]
    esm2_aligned = esm2_max[[esm2_idx[p] for p in shared]][:, esm2_active]

    log.info(f"Computing convergence: {n_active} x {len(esm2_active)}...")
    chunk_size = 500
    best_r = np.zeros(n_active)
    for start in range(0, n_active, chunk_size):
        end = min(start + chunk_size, n_active)
        chunk = esm3_aligned[:, start:end]
        chunk_mean = chunk.mean(axis=0, keepdims=True)
        chunk_std = chunk.std(axis=0, keepdims=True) + 1e-10
        chunk_norm = (chunk - chunk_mean) / chunk_std
        esm2_mean = esm2_aligned.mean(axis=0, keepdims=True)
        esm2_std = esm2_aligned.std(axis=0, keepdims=True) + 1e-10
        esm2_norm = (esm2_aligned - esm2_mean) / esm2_std
        corr = (chunk_norm.T @ esm2_norm) / len(shared)
        best_r[start:end] = corr.max(axis=1)
        if end % 2000 == 0 or end == n_active:
            log.info(f"  {end}/{n_active}")

    # 3. GO enrichment
    log.info("=== Loading GO enrichment ===")
    go = json.load(open(RESULTS / "go_enrichment.json"))
    go_map = {f["feature_id"]: f.get("n_enriched_terms", 0)
              for f in go["esm3"].get("per_feature_results", [])}

    # 4. Probe weights
    log.info("=== Loading probe weights ===")
    dg = json.load(open(RESULTS / "decoder_geometry.json"))
    probe_map = {m["feature_id"]: m.get("probe_weight", 0.0)
                 for m in dg["esm3"]["feature_metadata"]}

    # 5. PCA (from decoder weights)
    log.info("=== Computing PCA ===")
    from sklearn.decomposition import PCA
    ckpt = torch.load(str(ROOT / "models" / "sae_1.5M" / SAE_TAG / "best.pt"),
                       map_location="cpu", weights_only=False)
    W_dec = ckpt["model_state_dict"]["decoder.weight"].float().numpy()
    W_active = W_dec[:, active_indices].T
    pca = PCA(n_components=2)
    coords = pca.fit_transform(W_active)
    var_explained = pca.explained_variance_ratio_

    # 6. Merge everything
    log.info("=== Building merged dataset ===")
    cat_map = {0: "inactive", 1: "enhanced", 2: "suppressed", 3: "invariant"}
    merged = []
    for i, feat_idx in enumerate(active_indices):
        fid = int(feat_idx) + 1  # 1-indexed
        cat = cat_map[categories[feat_idx]]
        # Some features active in residue data but not in cross-modal test
        # (cross-modal only tested ~8900 proteins with structures)
        if cat == "inactive":
            cat = "untested"

        merged.append({
            "feature_id": fid,
            "pca_x": float(coords[i, 0]),
            "pca_y": float(coords[i, 1]),
            "activation_count": int(feat_counts[feat_idx]),
            "probe_weight": probe_map.get(fid, 0.0),
            "n_go_terms": go_map.get(fid, 0),
            "cross_modal_category": cat,
            "cohens_d": float(effect_sizes[feat_idx]),
            "mean_delta": float(mean_deltas[feat_idx]),
            "convergence_r": float(best_r[i]),
        })

    # Category counts
    cats = {}
    for m in merged:
        cats[m["cross_modal_category"]] = cats.get(m["cross_modal_category"], 0) + 1
    log.info(f"Categories: {cats}")

    output = {
        "model": "esm3",
        "method": "PCA",
        "variance_explained": [float(v) for v in var_explained],
        "n_active": len(merged),
        "category_counts": cats,
        "feature_metadata": merged,
    }

    out_path = RESULTS / "decoder_pca.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved {len(merged)} features to {out_path}")

    # Summary statistics by category
    for cat in ["enhanced", "suppressed", "invariant", "untested"]:
        feats = [m for m in merged if m["cross_modal_category"] == cat]
        if feats:
            conv = [m["convergence_r"] for m in feats]
            go_n = [m["n_go_terms"] for m in feats if m["n_go_terms"] > 0]
            log.info(f"  {cat}: n={len(feats)}, median_conv={np.median(conv):.3f}, "
                     f"GO-enriched={len(go_n)}/{len(feats)}")


if __name__ == "__main__":
    main()
