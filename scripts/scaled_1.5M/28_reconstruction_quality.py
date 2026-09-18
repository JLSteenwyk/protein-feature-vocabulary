#!/usr/bin/env python3
"""Eval-set reconstruction quality for SAEs.

Measures how well SAEs reconstruct held-out eval activations:
cosine similarity, MSE, variance explained, per-protein breakdown.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/28_reconstruction_quality.py
"""

import os
import sys
import json
import glob
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("recon")

MODEL_ROOT = ROOT / "models" / "sae_1.5M"
ACTIVATION_ROOT = ROOT / "results" / "scaled_1.5M" / "activations"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

BATCH_SIZE = 4096
MAX_RESIDUES = 500_000  # subsample for speed
SEED = 42


def load_sae(model_name, level="residue"):
    """Load the best SAE for a model."""
    from sae.model import build_sae

    if model_name == "esm3":
        tag = "esm3_residue_ef8_k64" if level == "residue" else "esm3_protein_ef8_k64"
    else:
        tag = "esm2_residue_ef8_k64" if level == "residue" else "esm2_protein_ef4_k64"

    ckpt_path = MODEL_ROOT / tag / "best.pt"
    if not ckpt_path.exists():
        log.warning(f"Checkpoint not found: {ckpt_path}")
        return None, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    config = ckpt["sae_config"]
    return sae, config


def evaluate_reconstruction(sae, activations, device="cpu"):
    """Compute reconstruction metrics on a batch of activations."""
    sae = sae.to(device)
    n = len(activations)

    all_cosine = []
    all_mse = []
    all_var_orig = []
    all_var_recon_err = []

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        x = torch.tensor(activations[start:end], dtype=torch.float32, device=device)

        with torch.no_grad():
            z = sae.encode(x)
            x_hat = sae.decode(z)

        # Cosine similarity
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
        all_cosine.append(cos.cpu().numpy())

        # MSE
        mse = ((x - x_hat) ** 2).mean(dim=-1)
        all_mse.append(mse.cpu().numpy())

        # For variance explained
        all_var_orig.append(x.var(dim=-1).cpu().numpy())
        all_var_recon_err.append((x - x_hat).var(dim=-1).cpu().numpy())

    cosines = np.concatenate(all_cosine)
    mses = np.concatenate(all_mse)
    var_orig = np.concatenate(all_var_orig)
    var_err = np.concatenate(all_var_recon_err)

    # Variance explained: 1 - var(error) / var(original)
    var_explained = 1 - var_err / (var_orig + 1e-10)

    return {
        "cosine_similarity": {
            "mean": float(np.mean(cosines)),
            "median": float(np.median(cosines)),
            "std": float(np.std(cosines)),
            "percentiles": {str(p): float(np.percentile(cosines, p))
                           for p in [5, 10, 25, 50, 75, 90, 95]},
        },
        "mse": {
            "mean": float(np.mean(mses)),
            "median": float(np.median(mses)),
            "std": float(np.std(mses)),
        },
        "variance_explained": {
            "mean": float(np.mean(var_explained)),
            "median": float(np.median(var_explained)),
            "std": float(np.std(var_explained)),
            "percentiles": {str(p): float(np.percentile(var_explained, p))
                           for p in [5, 10, 25, 50, 75, 90, 95]},
        },
        "n_samples": int(len(cosines)),
    }


def load_eval_activations(model_name, level="residue", max_residues=MAX_RESIDUES):
    """Load eval activations from chunked HDF5 files."""
    # Both models' eval activations are stored at L33
    layer_tag = "residue_L33" if level == "residue" else "protein_L33"

    act_dir = ACTIVATION_ROOT / model_name / layer_tag
    if not act_dir.exists():
        log.warning(f"Activation directory not found: {act_dir}")
        return None

    chunks = sorted(glob.glob(str(act_dir / "chunk*.h5")))
    if not chunks:
        log.warning(f"No chunk files in {act_dir}")
        return None

    all_acts = []
    total = 0
    for chunk_path in chunks:
        with h5py.File(chunk_path, "r") as f:
            acts = f["activations"][:].astype(np.float32)
            all_acts.append(acts)
            total += len(acts)
        if total >= max_residues:
            break

    combined = np.concatenate(all_acts, axis=0)
    if len(combined) > max_residues:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(combined), max_residues, replace=False)
        combined = combined[idx]

    return combined


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    results = {}

    for model_name in ["esm3", "esm2"]:
        log.info(f"\n{'='*60}")
        log.info(f"  {model_name.upper()} RECONSTRUCTION QUALITY")
        log.info(f"{'='*60}")

        model_results = {}

        for level in ["residue", "protein"]:
            log.info(f"\n--- {level}-level ---")

            sae, config = load_sae(model_name, level)
            if sae is None:
                log.warning(f"  Skipping {model_name} {level} (no SAE)")
                continue

            log.info(f"  SAE: {config.architecture}, d={config.dict_size}, k={getattr(config, 'k', 'N/A')}")

            max_n = MAX_RESIDUES if level == "residue" else 50_000
            acts = load_eval_activations(model_name, level, max_residues=max_n)
            if acts is None:
                log.warning(f"  Skipping {model_name} {level} (no activations)")
                continue

            log.info(f"  Loaded {len(acts):,} {level}-level activations, dim={acts.shape[1]}")

            metrics = evaluate_reconstruction(sae, acts, device=args.device)
            metrics["sae_tag"] = f"{model_name}_{level}"
            metrics["sae_architecture"] = config.architecture
            metrics["d_sae"] = config.dict_size
            metrics["d_model"] = config.input_dim

            log.info(f"  Cosine: {metrics['cosine_similarity']['mean']:.4f} "
                     f"(±{metrics['cosine_similarity']['std']:.4f})")
            log.info(f"  MSE: {metrics['mse']['mean']:.6f}")
            log.info(f"  Var explained: {metrics['variance_explained']['mean']:.4f}")

            model_results[level] = metrics

        results[model_name] = model_results

    out_path = OUTPUT_DIR / "reconstruction_quality.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
