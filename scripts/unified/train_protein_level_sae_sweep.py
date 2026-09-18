#!/usr/bin/env python3
"""Train multiple protein-level SAE configs, pick best, run autointerpretability.

Sweeps over expansion_factor and k to find a config with good reconstruction
AND enough discriminative active features for autointerpretability.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/train_protein_level_sae_sweep.py
"""

import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

from sae.model import TopKSAE, SAEConfig, build_sae
from sae.train import SAETrainer, TrainConfig

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prot_sae_sweep")


def evaluate_sae(sae, val_tensors):
    """Evaluate SAE quality. Returns dict of metrics."""
    sae.eval()
    with torch.no_grad():
        x = val_tensors[:500]
        z = sae.encode(x)
        x_hat = sae.decode(z)

        mse = ((x - x_hat) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=1).mean().item()
        x_var = x.var(dim=0).sum().item()
        res_var = (x - x_hat).var(dim=0).sum().item()
        var_explained = 1 - res_var / x_var

        # Active features and discrimination
        z_full = sae.encode(val_tensors)
        active_mask = (z_full > 0).any(dim=0)
        n_active = active_mask.sum().item()
        n_total = z_full.shape[1]
        dead_frac = 1 - n_active / n_total

        # Feature discrimination: for each active feature, compute CV of activations
        # Higher CV = more discriminative (activates strongly on some proteins, weakly on others)
        if n_active > 0:
            active_z = z_full[:, active_mask]
            # For each feature: std / mean among proteins where it's active
            cvs = []
            for j in range(active_z.shape[1]):
                vals = active_z[:, j]
                nonzero = vals[vals > 0]
                if len(nonzero) > 5:
                    cv = nonzero.std().item() / (nonzero.mean().item() + 1e-8)
                    cvs.append(cv)
            mean_cv = np.mean(cvs) if cvs else 0.0
            # Fraction of proteins each feature activates on (lower = more selective)
            active_fracs = (active_z > 0).float().mean(dim=0)
            mean_selectivity = 1 - active_fracs.mean().item()  # 1 = perfect selectivity
        else:
            mean_cv = 0.0
            mean_selectivity = 0.0

    return {
        "mse": mse,
        "cosine": cos,
        "var_explained": var_explained,
        "dead_frac": dead_frac,
        "n_active": n_active,
        "n_total": n_total,
        "mean_cv": mean_cv,
        "mean_selectivity": mean_selectivity,
    }


def score_config(metrics):
    """Score a config for autointerpretability suitability.

    Prioritizes: enough active features, good discrimination (high CV),
    good reconstruction (high cosine), and selectivity.
    """
    # Need at least 20 active features
    if metrics["n_active"] < 20:
        return -1.0

    # Weighted score
    score = (
        0.3 * metrics["cosine"]          # reconstruction quality
        + 0.3 * min(metrics["mean_cv"], 1.0)  # feature discrimination (cap at 1.0)
        + 0.2 * metrics["mean_selectivity"]    # feature selectivity
        + 0.2 * min(metrics["n_active"] / 200, 1.0)  # enough features (cap at 200)
    )
    return score


def main():
    # Load pre-computed protein-level activations (from previous run)
    h5_path = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33_protein_level.h5"
    if not h5_path.exists():
        log.error(f"Protein-level activations not found at {h5_path}. Run train_protein_level_sae.py first.")
        sys.exit(1)

    with h5py.File(str(h5_path), "r") as f:
        X = torch.tensor(f["activations"][:], dtype=torch.float32)
        all_accs = [a.decode() for a in f["accessions"][:]]
    d_model = X.shape[1]
    log.info(f"Loaded protein-level activations: {X.shape}")

    # Train/val split (same seed as original)
    n_train = int(0.9 * len(X))
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(42))
    train_tensors = X[perm[:n_train]]
    val_tensors = X[perm[n_train:]]
    log.info(f"Train: {train_tensors.shape}, Val: {val_tensors.shape}")

    # Sweep configs
    configs = [
        {"expansion_factor": 1, "k": 8,  "tag": "ef1_k8"},
        {"expansion_factor": 1, "k": 16, "tag": "ef1_k16"},
        {"expansion_factor": 1, "k": 32, "tag": "ef1_k32"},
        {"expansion_factor": 2, "k": 8,  "tag": "ef2_k8"},
        {"expansion_factor": 2, "k": 16, "tag": "ef2_k16"},
        {"expansion_factor": 2, "k": 32, "tag": "ef2_k32"},
        {"expansion_factor": 4, "k": 16, "tag": "ef4_k16"},
        {"expansion_factor": 4, "k": 32, "tag": "ef4_k32"},
    ]

    results = []

    for ci, cfg in enumerate(configs):
        tag = cfg["tag"]
        log.info(f"\n{'='*60}")
        log.info(f"Config {ci+1}/{len(configs)}: {tag} (expansion={cfg['expansion_factor']}, k={cfg['k']})")
        log.info(f"{'='*60}")

        sae_config = SAEConfig(
            input_dim=d_model,
            expansion_factor=cfg["expansion_factor"],
            k=cfg["k"],
            architecture="topk",
        )

        output_dir = str(ROOT / "models" / "sae" / "esm3" / f"S_St_layer_33_protein_level_{tag}")

        train_config = TrainConfig(
            lr=3e-4,
            batch_size=256,
            num_epochs=100,
            warmup_steps=100,
            log_every=50,
            save_every=500,
            output_dir=output_dir,
        )

        log.info(f"Dict size: {sae_config.input_dim * sae_config.expansion_factor}")

        trainer = SAETrainer(sae_config, train_config)
        trainer.train_on_tensor(train_tensors, val_tensors)

        # Load best checkpoint
        best_path = Path(output_dir) / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            sae = build_sae(ckpt["sae_config"])
            sae.load_state_dict(ckpt["model_state_dict"])
        else:
            sae = trainer.sae.cpu()

        metrics = evaluate_sae(sae, val_tensors)
        score = score_config(metrics)

        log.info(f"\nResults for {tag}:")
        log.info(f"  Cosine: {metrics['cosine']:.4f}")
        log.info(f"  Var explained: {100*metrics['var_explained']:.1f}%")
        log.info(f"  Active features: {metrics['n_active']}/{metrics['n_total']}")
        log.info(f"  Dead fraction: {100*metrics['dead_frac']:.1f}%")
        log.info(f"  Mean CV: {metrics['mean_cv']:.3f}")
        log.info(f"  Mean selectivity: {metrics['mean_selectivity']:.3f}")
        log.info(f"  SCORE: {score:.4f}")

        results.append({
            "tag": tag,
            "config": cfg,
            "metrics": metrics,
            "score": score,
            "output_dir": output_dir,
        })

        # Clean up GPU memory
        del trainer, sae
        torch.cuda.empty_cache()

    # Pick best
    results.sort(key=lambda r: r["score"], reverse=True)

    log.info(f"\n{'='*60}")
    log.info("SWEEP RESULTS (ranked by score)")
    log.info(f"{'='*60}")
    for r in results:
        log.info(f"  {r['tag']:12s} | score={r['score']:.4f} | "
                 f"cos={r['metrics']['cosine']:.3f} | "
                 f"active={r['metrics']['n_active']:4d} | "
                 f"CV={r['metrics']['mean_cv']:.3f} | "
                 f"sel={r['metrics']['mean_selectivity']:.3f}")

    best = results[0]
    log.info(f"\nBest config: {best['tag']} (score={best['score']:.4f})")

    # Copy best to canonical path
    best_src = Path(best["output_dir"]) / "best.pt"
    best_dst = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_protein_level_topk" / "best.pt"
    if best_src.exists():
        import shutil
        best_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, best_dst)
        log.info(f"Copied best model to {best_dst}")

    # Save sweep results
    sweep_out = ROOT / "results" / "unified" / "crossmodal_features" / "protein_level_sae_sweep.json"
    sweep_out.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved sweep results to {sweep_out}")

    log.info(f"\nBest config tag: {best['tag']}")
    log.info("Done! Run autointerpretability next.")


if __name__ == "__main__":
    main()
