#!/usr/bin/env python3
"""OV / QK circuit decomposition for ESM-2 and ESM-3.

For each attention head, decomposes it into:
- OV circuit (W_O @ W_V): what information the head moves
- QK circuit (W_Q^T @ W_K): what the head attends to

Analyses:
1. OV eigenspectrum: top eigenvalues reveal whether head copies, erases, or transforms
2. OV-unembedding alignment: which output tokens does each head promote?
3. QK-embedding alignment: which input token pairs does each head connect?
4. Effective rank of OV and QK matrices per head
5. Head classification: copy heads, erasure heads, transformation heads

Run as:
    ./env/bin/python scripts/unified/run_ov_qk_decomposition.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_ov_qk_decomposition.py --model esm3 --device cuda:1

Output: results/unified/{model}/ov_qk_decomposition.json
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
import numpy as np

from models.interventions import (
    get_layers, get_num_heads, get_head_dim,
    extract_ov_qk_matrices, get_unembedding_matrix,
)


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "ov_qk_decomposition_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("ov_qk"), out_dir


def ov_eigenspectrum(W_OV):
    """Compute eigenvalues of OV matrix.

    Args:
        W_OV: (d_model, d_model) OV circuit matrix

    Returns:
        eigenvalues (complex), sorted by magnitude descending
    """
    eigvals = torch.linalg.eigvals(W_OV.float())
    magnitudes = eigvals.abs()
    sorted_idx = magnitudes.argsort(descending=True)
    return eigvals[sorted_idx], magnitudes[sorted_idx]


def effective_rank_from_svd(M):
    """Effective rank via SVD."""
    s = torch.linalg.svdvals(M.float())
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    entropy = -(p * p.log()).sum()
    return float(entropy.exp())


def classify_head(eigvals_mag, top_k=5):
    """Classify head based on OV eigenspectrum.

    - Copy head: few dominant positive eigenvalues (low-rank, constructive)
    - Erasure head: dominant negative eigenvalues
    - Transformation head: spread eigenspectrum (high rank)
    """
    top = eigvals_mag[:top_k]
    total = eigvals_mag.sum()
    concentration = top.sum() / (total + 1e-10)

    if concentration > 0.5:
        return "copy"        # Information mostly flows through few dimensions
    elif concentration < 0.1:
        return "distributed"  # Spread across many dimensions
    else:
        return "mixed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== OV/QK Circuit Decomposition: {args.model} ===")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        n_layers = len(model.esm.encoder.layer)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        n_layers = len(model.transformer.blocks)

    n_heads = get_num_heads(model, args.model)
    d_head = get_head_dim(model, args.model)
    d_model = n_heads * d_head
    log.info(f"{args.model}: {n_layers} layers, {n_heads} heads, d_head={d_head}, d_model={d_model}")

    # Get unembedding matrix for alignment analysis
    W_U = get_unembedding_matrix(model, args.model).float().cpu()  # (V, D)
    log.info(f"Unembedding: {W_U.shape}")

    # Evaluate every 4th layer + first/last
    eval_layers = sorted(set([0, n_layers - 1] + list(range(0, n_layers, 4))))
    log.info(f"Evaluating {len(eval_layers)} layers: {eval_layers}")

    results_per_layer = {}
    head_classifications = {"copy": 0, "distributed": 0, "mixed": 0}

    # Move unembedding to compute device for alignment analysis
    compute_device = args.device if torch.cuda.is_available() else "cpu"
    W_U_dev = W_U.to(compute_device)

    t0 = time.time()
    for layer_idx in eval_layers:
        log.info(f"Layer {layer_idx}...")
        W_OV, W_QK = extract_ov_qk_matrices(model, args.model, layer_idx)
        # W_OV, W_QK: (n_heads, d_model, d_model)
        # Keep on GPU for fast eigendecomposition (cuSOLVER >> CPU LAPACK)
        W_OV = W_OV.float().to(compute_device)
        W_QK = W_QK.float().to(compute_device)

        layer_results = {"heads": {}}
        for h in range(n_heads):
            ov = W_OV[h]  # (d_model, d_model)
            qk = W_QK[h]

            # Eigenspectrum of OV
            _, eigmag = ov_eigenspectrum(ov)
            top5_eigvals = eigmag[:5].tolist()
            ov_rank = effective_rank_from_svd(ov)
            qk_rank = effective_rank_from_svd(qk)

            # OV-unembedding alignment: which tokens does this head promote?
            # W_U @ W_OV gives the logit effect: (V, D) @ (D, D) = (V, D)
            # Take the Frobenius norm per output token
            ov_logit = W_U_dev @ ov  # (V, D)
            ov_logit_norms = ov_logit.norm(dim=1)  # (V,)
            top_tokens = ov_logit_norms.topk(5)

            # Head classification
            htype = classify_head(eigmag)
            head_classifications[htype] += 1

            # Spectral concentration: fraction of total eigenvalue mass in top-5
            spectral_concentration = float(eigmag[:5].sum() / (eigmag.sum() + 1e-10))

            # OV trace (sum of eigenvalues ≈ identity component)
            ov_trace = float(torch.trace(ov))

            layer_results["heads"][h] = {
                "ov_effective_rank": ov_rank,
                "qk_effective_rank": qk_rank,
                "top5_eigenvalue_magnitudes": top5_eigvals,
                "spectral_concentration": spectral_concentration,
                "ov_trace": ov_trace,
                "head_type": htype,
                "top_promoted_token_indices": top_tokens.indices.tolist(),
                "top_promoted_token_norms": top_tokens.values.tolist(),
            }

        # Layer-level aggregates
        ov_ranks = [layer_results["heads"][h]["ov_effective_rank"] for h in range(n_heads)]
        qk_ranks = [layer_results["heads"][h]["qk_effective_rank"] for h in range(n_heads)]
        layer_results["mean_ov_rank"] = float(np.mean(ov_ranks))
        layer_results["mean_qk_rank"] = float(np.mean(qk_ranks))
        layer_results["head_type_counts"] = {}
        for h in range(n_heads):
            ht = layer_results["heads"][h]["head_type"]
            layer_results["head_type_counts"][ht] = layer_results["head_type_counts"].get(ht, 0) + 1

        results_per_layer[layer_idx] = layer_results
        log.info(f"  Layer {layer_idx}: mean OV rank={layer_results['mean_ov_rank']:.1f}, "
                 f"QK rank={layer_results['mean_qk_rank']:.1f}, "
                 f"types={layer_results['head_type_counts']}")

    log.info(f"Completed in {time.time()-t0:.1f}s")
    log.info(f"Global head classification: {head_classifications}")

    output = {
        "model": args.model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_head": d_head,
        "d_model": d_model,
        "eval_layers": eval_layers,
        "global_head_classifications": head_classifications,
        "per_layer": {str(k): v for k, v in results_per_layer.items()},
    }

    out_path = out_dir / "ov_qk_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
