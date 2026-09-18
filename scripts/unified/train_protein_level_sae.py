#!/usr/bin/env python3
"""Train a protein-level SAE on mean-pooled ESM-3 representations.

Mean-pools residue activations per protein, then trains a TopK SAE.
This replicates the protein-level approach from InterPLM/Gujral et al.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/train_protein_level_sae.py
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
log = logging.getLogger("prot_sae")


def main():
    # Load sequences for offsets
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)

    all_accs = sorted(sequences.keys())
    offsets = {}
    pos = 0
    for acc in all_accs:
        L = len(sequences[acc])
        offsets[acc] = (pos, L)
        pos += L

    # Mean-pool residue activations to protein level
    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33.h5")

    log.info(f"Mean-pooling {len(all_accs)} proteins...")
    protein_vectors = []
    valid_accs = []

    with h5py.File(h5_path, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]
        d_model = acts.shape[1]
        log.info(f"Activation shape: {acts.shape}")

        for i, acc in enumerate(all_accs):
            start, L = offsets[acc]
            if start + L > total or L < 5:
                continue
            chunk = acts[start:start + L]
            mean_vec = chunk.mean(axis=0)
            protein_vectors.append(mean_vec)
            valid_accs.append(acc)

            if (i + 1) % 1000 == 0:
                log.info(f"  Pooled {i + 1}/{len(all_accs)}")

    X = np.stack(protein_vectors).astype(np.float32)
    log.info(f"Protein-level matrix: {X.shape}")

    # Save protein-level activations
    out_h5 = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33_protein_level.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("activations", data=X)
        f.create_dataset("accessions", data=[a.encode() for a in valid_accs])
    log.info(f"Saved protein-level activations to {out_h5}")

    # Train SAE
    log.info("\nTraining protein-level SAE...")

    sae_config = SAEConfig(
        input_dim=d_model,
        expansion_factor=8,
        k=32,
        architecture="topk",
    )

    output_dir = str(ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_protein_level_topk")

    train_config = TrainConfig(
        lr=3e-4,
        batch_size=256,
        num_epochs=100,
        warmup_steps=100,
        log_every=50,
        save_every=500,
        output_dir=output_dir,
    )

    # Create torch tensor and split 90/10
    X_tensor = torch.tensor(X)
    n_train = int(0.9 * len(X_tensor))
    perm = torch.randperm(len(X_tensor), generator=torch.Generator().manual_seed(42))
    train_tensors = X_tensor[perm[:n_train]]
    val_tensors = X_tensor[perm[n_train:]]
    log.info(f"Train: {train_tensors.shape}, Val: {val_tensors.shape}")

    trainer = SAETrainer(sae_config, train_config)
    trainer.train_on_tensor(train_tensors, val_tensors)

    log.info("\nTraining complete!")

    # Evaluate — load best checkpoint
    best_path = Path(output_dir) / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        sae = build_sae(ckpt["sae_config"])
        sae.load_state_dict(ckpt["model_state_dict"])
        log.info(f"Loaded best checkpoint from {best_path}")
    else:
        sae = trainer.sae.cpu()
        log.info("No best checkpoint found, using final model")

    sae.eval()
    with torch.no_grad():
        x_sample = val_tensors[:500]
        z = sae.encode(x_sample)
        x_hat = sae.decode(z)

        mse = ((x_sample - x_hat) ** 2).mean().item()
        cos = torch.nn.functional.cosine_similarity(x_sample, x_hat, dim=1).mean().item()

        x_var = x_sample.var(dim=0).sum().item()
        res_var = (x_sample - x_hat).var(dim=0).sum().item()
        var_explained = 1 - res_var / x_var

        dead = (z.sum(dim=0) == 0).float().mean().item()

    log.info(f"Reconstruction quality:")
    log.info(f"  MSE: {mse:.4f}")
    log.info(f"  Cosine: {cos:.4f}")
    log.info(f"  Variance explained: {100*var_explained:.1f}%")
    log.info(f"  Dead features: {100*dead:.1f}%")


if __name__ == "__main__":
    main()
