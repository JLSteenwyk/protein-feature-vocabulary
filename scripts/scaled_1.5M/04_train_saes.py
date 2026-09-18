#!/usr/bin/env python3
"""Train SAEs on 1.5M protein activations (protein-level + residue-level).

Reads chunked HDF5 files from extraction step.
For residue-level: samples random subset to fit in memory, trains SAE.
For protein-level: loads all mean-pooled vectors (fits in memory easily).

Trains SAEs for both ESM-3 and ESM-2.

Usage:
    ./env/bin/python scripts/scaled_1.5M/04_train_saes.py --model esm3
    ./env/bin/python scripts/scaled_1.5M/04_train_saes.py --model esm2
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

from sae.model import SAEConfig, build_sae
from sae.train import SAETrainer, TrainConfig

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_saes")

STORAGE_ROOT = Path("/mnt/85740f55-8e9a-4214-9500-be446866627e/interpretability_1.5M")
MODEL_ROOT = ROOT / "models" / "sae_1.5M"

# Max residues to load for residue-level SAE training
# ESM-3: 1536 dims × 4 bytes × 20M = 122 GB; ESM-2: 1280 × 4 × 20M = 102 GB
MAX_RESIDUES = 20_000_000


def load_protein_level(model_name):
    """Load all protein-level mean-pooled vectors from chunks."""
    protein_dir = STORAGE_ROOT / model_name / "protein_L33"
    chunks = sorted(protein_dir.glob("*.h5"))
    log.info(f"Loading protein-level from {len(chunks)} chunk files...")

    all_acts = []
    all_ids = []
    for cp in chunks:
        with h5py.File(cp, "r") as f:
            all_acts.append(f["activations"][:])
            all_ids.extend([s.decode() for s in f["ids"][:]])

    X = np.concatenate(all_acts, axis=0).astype(np.float32)
    log.info(f"Protein-level: {X.shape}, {len(all_ids)} proteins")
    return X, all_ids


def load_residue_sample(model_name, max_residues=MAX_RESIDUES, seed=42):
    """Load a random sample of residues from chunks (reservoir sampling)."""
    residue_dir = STORAGE_ROOT / model_name / "residue_L33"
    chunks = sorted(residue_dir.glob("*.h5"))
    log.info(f"Sampling up to {max_residues/1e6:.0f}M residues from {len(chunks)} chunks...")

    rng = np.random.RandomState(seed)

    # First pass: count total residues
    total_residues = 0
    chunk_sizes = []
    for cp in chunks:
        with h5py.File(cp, "r") as f:
            n = f["activations"].shape[0]
            chunk_sizes.append(n)
            total_residues += n
    log.info(f"Total residues across chunks: {total_residues:,}")

    # Sampling fraction
    if total_residues <= max_residues:
        frac = 1.0
        log.info("Loading ALL residues (fits in budget)")
    else:
        frac = max_residues / total_residues
        log.info(f"Sampling fraction: {frac:.3f}")

    # Second pass: load sampled residues
    sampled = []
    n_loaded = 0
    for ci, cp in enumerate(chunks):
        with h5py.File(cp, "r") as f:
            acts = f["activations"][:]  # float16 array
            n = acts.shape[0]

            if frac < 1.0:
                n_sample = int(n * frac)
                if n_sample < 1:
                    n_sample = 1
                idx = rng.choice(n, n_sample, replace=False)
                acts = acts[idx]

            sampled.append(acts.astype(np.float32))
            n_loaded += acts.shape[0]

        if (ci + 1) % 20 == 0:
            log.info(f"  Loaded {ci+1}/{len(chunks)} chunks, {n_loaded:,} residues")

        if n_loaded >= max_residues:
            break

    X = np.concatenate(sampled, axis=0)
    log.info(f"Residue sample: {X.shape} ({X.nbytes / 1e9:.1f} GB)")
    return X


def train_sae(X, sae_type, model_name, d_model, expansion_factor, k, num_epochs,
              batch_size=4096, lr=3e-4, device="cuda"):
    """Train a single SAE and return metrics."""
    tag = f"{model_name}_{sae_type}_ef{expansion_factor}_k{k}"
    output_dir = str(MODEL_ROOT / tag)

    sae_config = SAEConfig(
        input_dim=d_model,
        expansion_factor=expansion_factor,
        k=k,
        architecture="topk",
    )

    train_config = TrainConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        warmup_steps=1000,
        log_every=500,
        save_every=5000,
        output_dir=output_dir,
        device=device,
    )

    log.info(f"\nTraining {tag}: dict_size={sae_config.input_dim * expansion_factor}, "
             f"k={k}, data={X.shape}")

    # Split 95/5 train/val
    n_val = max(int(0.05 * len(X)), 1000)
    perm = np.random.RandomState(42).permutation(len(X))
    X_train = torch.tensor(X[perm[n_val:]])
    X_val = torch.tensor(X[perm[:n_val]])

    trainer = SAETrainer(sae_config, train_config)
    summary = trainer.train_on_tensor(X_train, X_val)

    # Evaluate
    best_path = Path(output_dir) / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        sae = build_sae(ckpt["sae_config"])
        sae.load_state_dict(ckpt["model_state_dict"])
    else:
        sae = trainer.sae.cpu()

    sae.eval()
    with torch.no_grad():
        x_sample = X_val[:2000].to("cpu")
        z = sae.encode(x_sample)
        x_hat = sae.decode(z)
        cos = torch.nn.functional.cosine_similarity(x_sample, x_hat, dim=1).mean().item()
        x_var = x_sample.var(dim=0).sum().item()
        res_var = (x_sample - x_hat).var(dim=0).sum().item()
        var_exp = 1 - res_var / x_var
        n_active = (z > 0).any(dim=0).sum().item()
        n_total = z.shape[1]
        dead_frac = 1 - n_active / n_total

    metrics = {
        "tag": tag,
        "cosine": cos,
        "var_explained": var_exp,
        "n_active": n_active,
        "n_total": n_total,
        "dead_frac": dead_frac,
        **summary,
    }

    log.info(f"  Cosine: {cos:.4f}, Var explained: {100*var_exp:.1f}%, "
             f"Active: {n_active}/{n_total} ({100*dead_frac:.1f}% dead)")

    del trainer, sae, X_train, X_val
    torch.cuda.empty_cache()

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["esm3", "esm2"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-residue", action="store_true", help="Skip residue-level SAE")
    parser.add_argument("--skip-protein", action="store_true", help="Skip protein-level SAE")
    args = parser.parse_args()

    d_model = 1536 if args.model == "esm3" else 1280
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    # ═══════════════════════════════════════════════════
    # Protein-level SAEs
    # ═══════════════════════════════════════════════════
    if not args.skip_protein:
        log.info("\n" + "=" * 60)
        log.info("PROTEIN-LEVEL SAE TRAINING")
        log.info("=" * 60)

        X_prot, prot_ids = load_protein_level(args.model)

        # With 1.5M proteins, we can use larger dictionaries
        for ef, k in [(2, 32), (4, 32), (4, 64), (8, 64)]:
            m = train_sae(X_prot, "protein", args.model, d_model,
                          expansion_factor=ef, k=k, num_epochs=20,
                          batch_size=4096, device=args.device)
            all_metrics.append(m)

        del X_prot

    # ═══════════════════════════════════════════════════
    # Residue-level SAEs
    # ═══════════════════════════════════════════════════
    if not args.skip_residue:
        log.info("\n" + "=" * 60)
        log.info("RESIDUE-LEVEL SAE TRAINING")
        log.info("=" * 60)

        X_res = load_residue_sample(args.model, max_residues=MAX_RESIDUES)

        # Residue-level: expansion_factor=8, k=32 (standard from prior work)
        for ef, k in [(4, 32), (8, 32), (8, 64)]:
            m = train_sae(X_res, "residue", args.model, d_model,
                          expansion_factor=ef, k=k, num_epochs=5,
                          batch_size=4096, device=args.device)
            all_metrics.append(m)

        del X_res

    # ═══════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("SAE TRAINING SUMMARY")
    log.info("=" * 60)
    for m in all_metrics:
        log.info(f"  {m['tag']:40s} | cos={m['cosine']:.3f} | "
                 f"var={100*m['var_explained']:.0f}% | "
                 f"active={m['n_active']:>6d}/{m['n_total']:<6d} | "
                 f"dead={100*m['dead_frac']:.1f}%")

    # Save
    out_path = MODEL_ROOT / f"{args.model}_sae_training_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    log.info(f"\nSaved summary to {out_path}")


if __name__ == "__main__":
    main()
