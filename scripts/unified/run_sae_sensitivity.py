#!/usr/bin/env python3
"""SAE hyperparameter sensitivity analysis.

Tests how dictionary size, TopK k, and architecture choice affect
feature quality (reconstruction, sparsity, interpretability).

Usage:
    ./env/bin/python scripts/unified/run_sae_sensitivity.py --device cuda:0
"""

import sys
import os
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

from sae.model import SAEConfig, TopKSAE, build_sae
from sae.train import SAETrainer, TrainConfig

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sae_sens")


def evaluate_sae(sae, val_acts, sequences, dssp_data, metadata, residue_map,
                 device="cpu", batch_size=4096):
    """Evaluate SAE on held-out data."""
    sae.eval()
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

    total_mse = 0.0
    total_var = 0.0
    total_cos = 0.0
    total_l0 = 0.0
    n_batches = 0

    all_z = []
    N = len(val_acts)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = torch.tensor(val_acts[start:end], dtype=torch.float32).to(device)
        with torch.no_grad():
            z = sae.encode(x)
            x_hat = sae.decode(z)

        mse = ((x - x_hat) ** 2).mean().item()
        var = x.var().item()
        cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
        l0 = (z > 0).float().sum(dim=-1).mean().item()

        total_mse += mse
        total_var += var
        total_cos += cos
        total_l0 += l0
        n_batches += 1

        all_z.append(z.cpu().numpy())

    Z = np.concatenate(all_z, axis=0)
    n_alive = int((Z > 0).any(axis=0).sum())
    dict_size = Z.shape[1]
    dead_frac = 1.0 - n_alive / dict_size

    avg_mse = total_mse / n_batches
    avg_var = total_var / n_batches
    pct_var = 1.0 - avg_mse / (avg_var + 1e-10)

    # Quick interpretability check on top-200 alive features
    alive_idx = np.where((Z > 0).any(axis=0))[0]
    # Sort by total activation
    total_act = Z[:, alive_idx].sum(axis=0)
    top_features = alive_idx[np.argsort(-total_act)[:200]]

    n_aa_specific = 0
    n_ss_specific = 0

    # Build AA/SS vectors
    aa_vec = np.zeros(N, dtype=int)
    ss_vec = np.full(N, 2, dtype=int)
    for i, (acc, pos) in enumerate(residue_map[:N]):
        seq = sequences.get(acc, "")
        if pos < len(seq) and seq[pos] in aa_to_idx:
            aa_vec[i] = aa_to_idx[seq[pos]]
        dssp = dssp_data.get(acc, [])
        if dssp and pos < len(dssp):
            ss3 = dssp[pos].get("ss3", "C")
            ss_vec[i] = {"H": 0, "E": 1}.get(ss3, 2)

    aa_bg = np.bincount(aa_vec, minlength=20).astype(float)
    aa_bg /= aa_bg.sum() + 1e-10
    ss_bg = np.bincount(ss_vec, minlength=3).astype(float)
    ss_bg /= ss_bg.sum() + 1e-10

    for fi in top_features:
        col = Z[:, fi]
        active = col > 0
        if active.sum() < 10:
            continue
        thresh = np.percentile(col[active], 95)
        top_mask = col >= thresh
        if top_mask.sum() < 5:
            continue

        aa_top = np.bincount(aa_vec[top_mask], minlength=20).astype(float)
        aa_top /= aa_top.sum() + 1e-10
        if (aa_top / (aa_bg + 1e-10)).max() > 4.0:
            n_aa_specific += 1

        ss_top = np.bincount(ss_vec[top_mask], minlength=3).astype(float)
        ss_top /= ss_top.sum() + 1e-10
        if (ss_top / (ss_bg + 1e-10)).max() > 1.5:
            n_ss_specific += 1

    return {
        "recon_mse": float(avg_mse),
        "pct_variance_explained": float(pct_var),
        "cosine_sim": float(total_cos / n_batches),
        "l0": float(total_l0 / n_batches),
        "n_alive": n_alive,
        "dead_frac": float(dead_frac),
        "dict_size": dict_size,
        "n_aa_specific_top200": n_aa_specific,
        "n_ss_specific_top200": n_ss_specific,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-residues", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "unified" / "sae_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json") as f:
        dssp_data = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    h5_path = ROOT / "data" / "activations" / "esm2_scaled" / "layer_24.h5"
    d_model = 1280

    # Load and split data
    log.info(f"Loading {args.n_residues} residues...")
    with h5py.File(str(h5_path), "r") as f:
        total = f["activations"].shape[0]
        rng = np.random.RandomState(42)
        idx = rng.choice(total, min(args.n_residues, total), replace=False)
        idx.sort()
        X = f["activations"][idx].astype(np.float32)

    # Build residue map
    accs = sorted(sequences.keys())
    offsets = []
    cur = 0
    for acc in accs:
        L = len(sequences[acc])
        offsets.append((acc, cur, cur + L))
        cur += L

    residue_map = []
    oi = 0
    for gi in idx:
        while oi < len(offsets) - 1 and gi >= offsets[oi + 1][1]:
            oi += 1
        acc, start, end = offsets[oi]
        residue_map.append((acc, gi - start))

    # Split 80/20
    n_train = int(0.8 * len(X))
    X_train = X[:n_train]
    X_val = X[n_train:]
    val_map = residue_map[n_train:]
    log.info(f"Train: {n_train}, Val: {len(X_val)}")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    val_tensor = torch.tensor(X_val, dtype=torch.float32)

    # ═══ Experiment Grid ═══
    experiments = []

    # 1. Dictionary size sweep (expansion_factor)
    for ef in [1, 2, 4, 8]:
        experiments.append({
            "name": f"topk_ef{ef}_k64",
            "config": SAEConfig(input_dim=d_model, expansion_factor=ef, k=64, architecture="topk"),
        })

    # 2. TopK k sweep
    for k in [16, 32, 64, 128]:
        experiments.append({
            "name": f"topk_ef8_k{k}",
            "config": SAEConfig(input_dim=d_model, expansion_factor=8, k=k, architecture="topk"),
        })

    # 3. Architecture comparison
    for l1 in [0.01, 0.05, 0.1]:
        experiments.append({
            "name": f"vanilla_ef8_l1_{l1}",
            "config": SAEConfig(input_dim=d_model, expansion_factor=8, architecture="vanilla",
                                l1_coeff=l1),
        })

    # Deduplicate (topk_ef8_k64 appears twice)
    seen = set()
    unique_experiments = []
    for exp in experiments:
        if exp["name"] not in seen:
            seen.add(exp["name"])
            unique_experiments.append(exp)
    experiments = unique_experiments

    log.info(f"\n{len(experiments)} experiments to run:")
    for exp in experiments:
        c = exp["config"]
        log.info(f"  {exp['name']}: dict_size={c.dict_size}, arch={c.architecture}")

    all_results = []

    for i, exp in enumerate(experiments):
        name = exp["name"]
        config = exp["config"]
        log.info(f"\n{'='*50}")
        log.info(f"[{i+1}/{len(experiments)}] {name}")
        log.info(f"{'='*50}")

        train_config = TrainConfig(
            lr=3e-4,
            batch_size=4096,
            num_epochs=args.epochs,
            warmup_steps=500,
            device=args.device,
            output_dir=str(out_dir / "checkpoints" / name),
            log_every=200,
        )

        try:
            trainer = SAETrainer(config, train_config)
            summary = trainer.train_on_tensor(train_tensor, val_tensor)

            # Evaluate
            trainer.sae.to("cpu").eval()
            eval_result = evaluate_sae(
                trainer.sae, X_val, sequences, dssp_data, metadata,
                val_map, device="cpu"
            )
            eval_result["name"] = name
            eval_result["architecture"] = config.architecture
            eval_result["expansion_factor"] = config.expansion_factor
            eval_result["k"] = config.k if config.architecture == "topk" else None
            eval_result["l1_coeff"] = config.l1_coeff if config.architecture == "vanilla" else None
            eval_result["train_loss"] = summary["final_loss"]
            eval_result["train_recon"] = summary["final_recon"]
            eval_result["val_loss"] = summary.get("best_val_loss")

            all_results.append(eval_result)

            log.info(f"  MSE={eval_result['recon_mse']:.4f}, "
                     f"VarExpl={eval_result['pct_variance_explained']:.3f}, "
                     f"L0={eval_result['l0']:.1f}, "
                     f"Dead={eval_result['dead_frac']:.3f}, "
                     f"AA={eval_result['n_aa_specific_top200']}, "
                     f"SS={eval_result['n_ss_specific_top200']}")

            # Free GPU memory
            del trainer
            torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"  Failed: {e}")
            all_results.append({"name": name, "error": str(e)})

    # Summary table
    log.info(f"\n{'='*80}")
    log.info(f"{'Name':<25} {'MSE':>8} {'Var%':>7} {'L0':>6} {'Dead%':>6} {'AA':>4} {'SS':>4}")
    log.info("-" * 80)
    for r in all_results:
        if "error" in r:
            log.info(f"{r['name']:<25} ERROR")
            continue
        log.info(f"{r['name']:<25} {r['recon_mse']:>8.4f} {r['pct_variance_explained']:>6.3f} "
                 f"{r['l0']:>6.1f} {r['dead_frac']*100:>5.1f}% "
                 f"{r['n_aa_specific_top200']:>4} {r['n_ss_specific_top200']:>4}")

    out_path = out_dir / "sae_sensitivity_L24.json"
    with open(out_path, "w") as f:
        json.dump({"experiments": all_results}, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
