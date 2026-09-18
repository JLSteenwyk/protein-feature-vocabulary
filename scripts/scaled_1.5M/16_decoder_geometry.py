#!/usr/bin/env python3
"""Decoder geometry analysis: UMAP of SAE decoder weights, cosine similarity structure.

Usage:
    ./env/bin/python scripts/scaled_1.5M/16_decoder_geometry.py
"""

import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

from sae.model import build_sae

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("decoder_geom")

MODEL_ROOT = ROOT / "models" / "sae_1.5M"
FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

CONFIGS = {
    "esm3": "esm3_residue_ef8_k64",
    "esm2": "esm2_residue_ef8_k64",
}


def run_decoder_geometry(model_name, sae_tag):
    """Analyze decoder geometry for one SAE."""
    log.info(f"\n{'='*60}")
    log.info(f"DECODER GEOMETRY: {model_name} ({sae_tag})")
    log.info(f"{'='*60}")

    # Load SAE
    path = MODEL_ROOT / sae_tag / "best.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    # Extract decoder weights: (d_model, d_sae) -> transpose to (d_sae, d_model)
    W_dec = sae.decoder.weight.detach().numpy().T  # (d_sae, d_model)
    d_sae, d_model = W_dec.shape
    log.info(f"  Decoder weights: {W_dec.shape}")

    # Get active features
    feat_dir = FEATURE_ROOT / model_name / "residue"
    with h5py.File(feat_dir / "protein_summaries.h5", "r") as f:
        feature_counts = f["feature_activation_counts"][:]

    active_mask = feature_counts > 0
    active_idx = np.where(active_mask)[0]
    W_active = W_dec[active_idx]
    log.info(f"  Active features: {len(active_idx)}")

    # Normalize decoder vectors
    norms = np.linalg.norm(W_active, axis=1, keepdims=True)
    norms[norms == 0] = 1
    W_normed = W_active / norms

    # Pairwise cosine similarity
    log.info("  Computing pairwise cosine similarity...")
    cos_sim = W_normed @ W_normed.T
    upper_tri = cos_sim[np.triu_indices(len(active_idx), k=1)]

    log.info(f"  Mean cosine similarity: {upper_tri.mean():.4f}")
    log.info(f"  Std cosine similarity: {upper_tri.std():.4f}")
    log.info(f"  Max cosine similarity: {upper_tri.max():.4f}")

    cos_percentiles = {str(p): float(np.percentile(upper_tri, p))
                       for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]}

    # UMAP embedding
    log.info("  Computing UMAP embedding...")
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine",
                            random_state=42, n_components=2)
        embedding = reducer.fit_transform(W_active)
        umap_coords = embedding.tolist()
        log.info(f"  UMAP done: {embedding.shape}")
    except ImportError:
        log.warning("  umap-learn not installed, skipping UMAP")
        umap_coords = None

    # Feature metadata for coloring
    feature_metadata = []
    # Load probe weights if available
    probe_path = OUTPUT_DIR / "interpretable_probing.json"
    probe_weights = {}
    if probe_path.exists():
        with open(probe_path) as f:
            probe_data = json.load(f)
        if model_name in probe_data:
            for feat in probe_data[model_name].get("top_features", []):
                probe_weights[feat["feature_id"]] = feat["weight"]

    # Load GO enrichment if available
    go_path = OUTPUT_DIR / "go_enrichment.json"
    go_enriched = {}
    if go_path.exists():
        with open(go_path) as f:
            go_data = json.load(f)
        if model_name in go_data:
            for feat_result in go_data[model_name].get("per_feature_results", []):
                fid = feat_result["feature_id"]
                if feat_result.get("top_terms"):
                    top_term = feat_result["top_terms"][0]["go_id"]
                    go_enriched[fid] = top_term

    for i, fid in enumerate(active_idx):
        fid = int(fid)
        meta = {
            "feature_id": fid,
            "activation_count": int(feature_counts[fid]),
            "decoder_norm": float(norms[i, 0]),
            "probe_weight": probe_weights.get(fid, 0.0),
            "has_go_enrichment": fid in go_enriched,
        }
        if umap_coords:
            meta["umap_x"] = umap_coords[i][0]
            meta["umap_y"] = umap_coords[i][1]
        feature_metadata.append(meta)

    return {
        "model": model_name,
        "sae_tag": sae_tag,
        "d_sae": d_sae,
        "d_model": d_model,
        "n_active": len(active_idx),
        "cosine_similarity": {
            "mean": float(upper_tri.mean()),
            "std": float(upper_tri.std()),
            "min": float(upper_tri.min()),
            "max": float(upper_tri.max()),
            "percentiles": cos_percentiles,
        },
        "has_umap": umap_coords is not None,
        "feature_metadata": feature_metadata[:500],  # cap for JSON size
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_name, sae_tag in CONFIGS.items():
        if not (MODEL_ROOT / sae_tag / "best.pt").exists():
            log.warning(f"No SAE for {model_name}")
            continue
        results[model_name] = run_decoder_geometry(model_name, sae_tag)

    out_path = OUTPUT_DIR / "decoder_geometry.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
