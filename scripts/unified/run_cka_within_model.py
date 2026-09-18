#!/usr/bin/env python3
"""Within-model CKA analysis for ESM-2 and ESM-3.

CPU-only. Loads pre-extracted activations from HDF5 files and computes
CKA between all pairs of extraction layers.

Run as:
    ./env/bin/python scripts/unified/run_cka_within_model.py --model esm2
    ./env/bin/python scripts/unified/run_cka_within_model.py --model esm3

Output: results/unified/{model}/cka_within_model.json
"""

import sys
import os
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from models.metrics import linear_cka, effective_rank


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "cka_within_model_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("cka_within"), out_dir


def load_esm_activations(model_name, max_proteins=200, max_residues=50000):
    """Load ESM-2 or ESM-3 activations from HDF5 files.

    Supports two formats:
    1. Per-layer files: layer_0.h5, layer_4.h5, ... with 'activations' key (flat residue matrix)
    2. Per-protein files: with 'residual_stream/{layer}' groups
    """
    import h5py

    if model_name in ("esm2",):
        act_dir = ROOT / "data" / "activations" / "esm2_scaled"
    elif model_name in ("esm3",):
        act_dir = ROOT / "data" / "activations" / "esm3_scaled"
    else:
        act_dir = ROOT / "data" / "activations" / model_name

    if not act_dir.exists():
        act_dir = ROOT / "data" / "activations" / model_name

    h5_files = sorted(act_dir.glob("*.h5"))

    # Detect format: per-layer files (layer_N.h5) or per-protein files
    per_layer_files = [f for f in h5_files if f.stem.startswith("layer_")]

    if per_layer_files:
        # Format 1: Per-layer files with flat 'activations' matrix
        layers = sorted(int(f.stem.split("_")[1]) for f in per_layer_files)
        activations = {}
        for layer in layers:
            h5_path = act_dir / f"layer_{layer}.h5"
            with h5py.File(h5_path, "r") as f:
                data = f["activations"]
                n_total = data.shape[0]
                n_use = min(n_total, max_residues)
                if n_use < n_total:
                    # Read contiguous blocks from spread-out positions (fast HDF5 access)
                    n_blocks = 10
                    block_size = n_use // n_blocks
                    block_starts = np.linspace(0, n_total - block_size, n_blocks, dtype=int)
                    chunks = []
                    for start in block_starts:
                        chunks.append(data[start:start + block_size].astype(np.float64))
                    arr = np.concatenate(chunks, axis=0)[:n_use]
                else:
                    arr = data[:].astype(np.float64)
                activations[layer] = arr
        return activations, layers
    else:
        # Format 2: Per-protein files with residual_stream/{layer} or trunk/single/{layer}
        h5_files = h5_files[:max_proteins]
        with h5py.File(h5_files[0], "r") as f:
            if "residual_stream" in f:
                layers = sorted(int(k) for k in f["residual_stream"].keys())
            elif "trunk/single" in f:
                layers = sorted(int(k) for k in f["trunk/single"].keys())
            else:
                raise ValueError(f"Unknown HDF5 format in {h5_files[0]}")

        all_data = {l: [] for l in layers}
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                for l in layers:
                    key = str(l)
                    if "residual_stream" in f and key in f["residual_stream"]:
                        data = f["residual_stream"][key][:]
                    elif "trunk/single" in f and key in f["trunk/single"]:
                        data = f["trunk/single"][key][:]
                    else:
                        continue
                    if data.ndim == 3:
                        data = data[0]
                    all_data[l].append(data)

        # Concatenate and subsample
        activations = {}
        for l in layers:
            cat = np.concatenate(all_data[l], axis=0).astype(np.float64)
            n_total = cat.shape[0]
            n_use = min(n_total, max_residues)
            if n_use < n_total:
                indices = np.random.default_rng(42).choice(n_total, n_use, replace=False)
                indices.sort()
                cat = cat[indices]
            activations[l] = cat
        return activations, layers



def compute_pairwise_cka(activations, layers, log, max_residues=5000):
    """Compute CKA between all pairs of layers.

    Accepts activations as either:
    - dict of lists of arrays (per-protein format) -> concatenates
    - dict of 2D arrays (per-layer format) -> uses directly
    Subsamples to max_residues for efficiency.
    """
    # Build per-layer matrices (N_residues, D)
    layer_matrices = {}
    for l in layers:
        if isinstance(activations[l], list):
            all_residues = np.concatenate(activations[l], axis=0)
        else:
            all_residues = activations[l]
        # Subsample if too many
        if all_residues.shape[0] > max_residues:
            rng = np.random.RandomState(42)
            idx = rng.choice(all_residues.shape[0], max_residues, replace=False)
            all_residues = all_residues[idx]
        layer_matrices[l] = all_residues.astype(np.float64)
        log.info(f"  Layer {l}: {all_residues.shape}")

    n = len(layers)
    cka_matrix = np.zeros((n, n))
    ranks = {}

    for i, li in enumerate(layers):
        ranks[li] = effective_rank(layer_matrices[li])
        for j, lj in enumerate(layers):
            if j < i:
                cka_matrix[i, j] = cka_matrix[j, i]  # Symmetric
            else:
                cka_matrix[i, j] = linear_cka(layer_matrices[li], layer_matrices[lj])

    return cka_matrix, ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--max-residues", type=int, default=5000)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== CKA Within-Model: {args.model} ===")

    t0 = time.time()
    activations, layers = load_esm_activations(args.model, max_proteins=args.max_proteins)

    # Compute residue count (works for both list-of-arrays and single-array formats)
    sample_val = next(iter(activations.values()))
    if isinstance(sample_val, list):
        n_residues = sum(a.shape[0] for a in sample_val)
    else:
        n_residues = sample_val.shape[0]
    log.info(f"Loaded {args.model}: {len(layers)} layers, {n_residues} residues per layer")

    cka_matrix, ranks = compute_pairwise_cka(activations, layers, log, max_residues=args.max_residues)
    log.info(f"CKA computation done in {time.time()-t0:.1f}s")

    results = {
        "model": args.model,
        "layers": layers,
        "cka_matrix": cka_matrix.tolist(),
        "effective_ranks": {str(k): v for k, v in ranks.items()},
        "n_residues": n_residues,
    }

    out_path = out_dir / "cka_within_model.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
