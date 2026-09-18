#!/usr/bin/env python3
"""Train a new S+St SAE on the scaled multimodal activations.

The original S+St SAE was trained on only 46K residues and has poor reconstruction
(cosine=0.20, variance=-320%). This retrains on the full scaled dataset (1.4M residues)
with the same hyperparameters as the production SAEs.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/train_sst_sae_scaled.py
"""

import sys
import os
import torch
import h5py
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from sae.model import SAEConfig
from sae.train import SAETrainer, TrainConfig

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_sst")


def main():
    h5_path = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33.h5"
    output_dir = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_scaled_topk"

    # Load activations
    log.info(f"Loading activations from {h5_path}")
    with h5py.File(h5_path, "r") as f:
        total = f["activations"].shape[0]
        d_model = f["activations"].shape[1]
        log.info(f"  Total residues: {total}, d_model: {d_model}")

        # 80/20 train/val split
        val_start = int(total * 0.8)
        train_acts = torch.tensor(f["activations"][:val_start], dtype=torch.float32)
        val_acts = torch.tensor(f["activations"][val_start:], dtype=torch.float32)

    log.info(f"  Train: {len(train_acts)}, Val: {len(val_acts)}")
    log.info(f"  Train mean: {train_acts.mean():.4f}, std: {train_acts.std():.4f}")
    log.info(f"  Train L2 norm: {torch.norm(train_acts, dim=1).mean():.2f}")

    # Same config as production SAEs
    sae_config = SAEConfig(
        input_dim=d_model,
        expansion_factor=8,
        k=64,
        architecture="topk",
        l1_coeff=0.001,
        tied_weights=False,
    )

    train_config = TrainConfig(
        lr=3e-4,
        batch_size=4096,
        num_epochs=15,
        warmup_steps=500,
        weight_decay=0.0,
        log_every=200,
        save_every=1000,
        device="cuda:0",
        output_dir=str(output_dir),
        normalize_decoder=True,
    )

    log.info(f"SAE config: {sae_config}")
    log.info(f"Dict size: {sae_config.input_dim * sae_config.expansion_factor}")
    log.info(f"Training for {train_config.num_epochs} epochs...")

    trainer = SAETrainer(sae_config, train_config)
    summary = trainer.train_on_tensor(train_acts, val_acts)

    log.info(f"\nTraining complete:")
    log.info(f"  Final loss: {summary['final_loss']:.6f}")
    log.info(f"  Final recon: {summary['final_recon']:.6f}")
    log.info(f"  Final L0: {summary['final_l0']:.1f}")
    log.info(f"  Dead frac: {summary['dead_frac']:.4f}")
    log.info(f"  Best val loss: {summary['best_val_loss']:.6f}")

    # Quick reconstruction quality check
    log.info("\nReconstruction quality check on validation set:")
    sae = trainer.sae.eval()
    with torch.no_grad():
        x = val_acts[:5000].to(train_config.device)
        z = sae.encode(x)
        x_hat = sae.decode(z)
        mse = ((x - x_hat)**2).mean().item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
        var_x = x.var(dim=0).sum().item()
        pct_var = 1 - ((x - x_hat)**2).sum().item() / x.shape[0] / (var_x + 1e-10)

    log.info(f"  MSE: {mse:.4f}")
    log.info(f"  Cosine similarity: {cos:.4f}")
    log.info(f"  Variance explained: {pct_var:.4f}")
    log.info(f"\nSaved to {output_dir}/best.pt")


if __name__ == "__main__":
    main()
