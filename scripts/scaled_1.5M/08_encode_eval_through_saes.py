#!/usr/bin/env python3
"""Encode eval protein activations through best SAEs to get feature matrices.

Reads HDF5 chunks from 07_extract_eval_activations.py, encodes through SAEs,
saves sparse feature activations and summary statistics.

Usage:
    ./env/bin/python scripts/scaled_1.5M/08_encode_eval_through_saes.py
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
from scipy import sparse

from sae.model import build_sae

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("encode_eval")

ACT_ROOT = ROOT / "results" / "scaled_1.5M" / "activations"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
OUTPUT_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"

# Best SAE configs from training
BEST_CONFIGS = {
    "esm3_residue": "esm3_residue_ef8_k64",
    "esm2_residue": "esm2_residue_ef8_k64",
    "esm3_protein": "esm3_protein_ef8_k64",
    "esm2_protein": "esm2_protein_ef8_k64",
}


def load_sae(tag):
    """Load SAE from checkpoint."""
    path = MODEL_ROOT / tag / "best.pt"
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae, ckpt["sae_config"]


def encode_residue_level(model_name, sae_tag):
    """Encode residue-level activations through SAE, save sparse features."""
    log.info(f"\n{'='*60}")
    log.info(f"Encoding {model_name} residue-level through {sae_tag}")
    log.info(f"{'='*60}")

    sae, config = load_sae(sae_tag)
    d_sae = config.input_dim * config.expansion_factor

    residue_dir = ACT_ROOT / model_name / "residue_L33"
    chunks = sorted(residue_dir.glob("*.h5"))
    log.info(f"  {len(chunks)} activation chunks")

    out_dir = OUTPUT_ROOT / model_name / "residue"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_protein_ids = []
    all_protein_offsets = []  # (start_residue, n_residues) for each protein
    global_offset = 0

    # Per-feature statistics
    feature_activation_counts = np.zeros(d_sae, dtype=np.int64)  # how many residues activate
    feature_activation_sums = np.zeros(d_sae, dtype=np.float64)

    # Per-protein max/mean feature activations (for GO enrichment)
    protein_max_acts = []
    protein_mean_acts = []

    # Sparse feature storage — accumulate COO data
    all_rows = []
    all_cols = []
    all_vals = []
    total_residues = 0

    for ci, chunk_path in enumerate(chunks):
        with h5py.File(chunk_path, "r") as f:
            acts = torch.tensor(f["activations"][:].astype(np.float32))
            offsets = f["offsets"][:]
            ids = [s.decode() for s in f["ids"][:]]

        # Encode through SAE in batches
        batch_size = 4096
        z_chunks = []
        for i in range(0, acts.shape[0], batch_size):
            batch = acts[i:i + batch_size]
            with torch.no_grad():
                z = sae.encode(batch).numpy()
            z_chunks.append(z)
        Z = np.concatenate(z_chunks, axis=0)

        # Per-protein processing
        for pi, pid in enumerate(ids):
            _, local_offset, L = offsets[pi]
            z_protein = Z[local_offset:local_offset + L]  # (L, d_sae)

            # Sparse storage
            rows, cols = np.nonzero(z_protein)
            vals = z_protein[rows, cols]
            all_rows.append(rows + global_offset)
            all_cols.append(cols)
            all_vals.append(vals.astype(np.float32))

            # Feature statistics
            active_mask = z_protein > 0
            feature_activation_counts += active_mask.sum(axis=0)
            feature_activation_sums += z_protein.sum(axis=0)

            # Per-protein summary
            if L > 0:
                protein_max_acts.append(z_protein.max(axis=0))
                protein_mean_acts.append(z_protein.mean(axis=0))
            else:
                protein_max_acts.append(np.zeros(d_sae))
                protein_mean_acts.append(np.zeros(d_sae))

            all_protein_ids.append(pid)
            all_protein_offsets.append((global_offset, L))
            global_offset += L

        total_residues += Z.shape[0]

        if (ci + 1) % 2 == 0 or ci == len(chunks) - 1:
            log.info(f"    Chunk {ci+1}/{len(chunks)}: {len(all_protein_ids)} proteins, "
                     f"{total_residues} residues")

    # Build sparse CSR matrix
    log.info("  Building sparse matrix...")
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.concatenate(all_vals)
    X_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(total_residues, d_sae))
    log.info(f"  Sparse matrix: {X_sparse.shape}, nnz={X_sparse.nnz:,} "
             f"({100*X_sparse.nnz/(X_sparse.shape[0]*X_sparse.shape[1]):.2f}% fill)")

    # Save sparse matrix
    sparse.save_npz(str(out_dir / "features_sparse.npz"), X_sparse)

    # Save protein offsets
    protein_max_arr = np.stack(protein_max_acts)
    protein_mean_arr = np.stack(protein_mean_acts)
    offsets_arr = np.array(all_protein_offsets, dtype=np.int64)

    with h5py.File(out_dir / "protein_summaries.h5", "w") as f:
        f.create_dataset("protein_ids", data=[s.encode() for s in all_protein_ids])
        f.create_dataset("offsets", data=offsets_arr)
        f.create_dataset("protein_max_activations", data=protein_max_arr.astype(np.float32))
        f.create_dataset("protein_mean_activations", data=protein_mean_arr.astype(np.float32))
        f.create_dataset("feature_activation_counts", data=feature_activation_counts)
        f.create_dataset("feature_activation_sums", data=feature_activation_sums)

    # Feature summary
    n_active = (feature_activation_counts > 0).sum()
    log.info(f"  Active features: {n_active}/{d_sae}")
    log.info(f"  Saved to {out_dir}/")

    return {
        "n_proteins": len(all_protein_ids),
        "n_residues": total_residues,
        "n_active_features": int(n_active),
        "d_sae": d_sae,
        "nnz": int(X_sparse.nnz),
    }


def encode_protein_level(model_name, sae_tag):
    """Encode protein-level mean-pooled activations through SAE."""
    log.info(f"\n{'='*60}")
    log.info(f"Encoding {model_name} protein-level through {sae_tag}")
    log.info(f"{'='*60}")

    sae, config = load_sae(sae_tag)
    d_sae = config.input_dim * config.expansion_factor

    protein_dir = ACT_ROOT / model_name / "protein_L33"
    chunks = sorted(protein_dir.glob("*.h5"))

    # Load all protein means
    all_acts = []
    all_ids = []
    for cp in chunks:
        with h5py.File(cp, "r") as f:
            all_acts.append(f["activations"][:])
            all_ids.extend([s.decode() for s in f["ids"][:]])

    X = np.concatenate(all_acts, axis=0).astype(np.float32)
    log.info(f"  Protein-level: {X.shape}")

    # Encode
    with torch.no_grad():
        Z = sae.encode(torch.tensor(X)).numpy()

    n_active = (Z > 0).any(axis=0).sum()
    log.info(f"  Active features: {n_active}/{d_sae}")

    # Save
    out_dir = OUTPUT_ROOT / model_name / "protein"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_dir / "features.h5", "w") as f:
        f.create_dataset("features", data=Z.astype(np.float32))
        f.create_dataset("protein_ids", data=[s.encode() for s in all_ids])

    log.info(f"  Saved to {out_dir}/")

    return {
        "n_proteins": len(all_ids),
        "n_active_features": int(n_active),
        "d_sae": d_sae,
    }


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for model_name in ["esm3", "esm2"]:
        # Check if activations exist
        if not (ACT_ROOT / model_name / "residue_L33").exists():
            log.warning(f"  No activations for {model_name}, skipping")
            continue

        # Residue-level
        res_tag = BEST_CONFIGS[f"{model_name}_residue"]
        if (MODEL_ROOT / res_tag / "best.pt").exists():
            summary[f"{model_name}_residue"] = encode_residue_level(model_name, res_tag)
        else:
            log.warning(f"  No SAE checkpoint for {res_tag}")

        # Protein-level
        prot_tag = BEST_CONFIGS[f"{model_name}_protein"]
        if (MODEL_ROOT / prot_tag / "best.pt").exists():
            summary[f"{model_name}_protein"] = encode_protein_level(model_name, prot_tag)
        else:
            log.warning(f"  No SAE checkpoint for {prot_tag}")

    # Save summary
    out_path = OUTPUT_ROOT / "encoding_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary saved to {out_path}")
    for k, v in summary.items():
        log.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
