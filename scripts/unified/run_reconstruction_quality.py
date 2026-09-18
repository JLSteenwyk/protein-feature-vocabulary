#!/usr/bin/env python3
"""Assess reconstruction quality of production SAEs.

Measures MSE, % variance explained, cosine similarity, L0 sparsity,
and analyzes which residues have worst reconstruction.

Usage:
    ./env/bin/python scripts/unified/run_reconstruction_quality.py
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

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("recon_qual")


def load_sae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def assess_reconstruction(sae, h5_path, sequences, dssp_data, metadata,
                           batch_size=4096, holdout_frac=0.2):
    """Full reconstruction quality assessment."""
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

    # Build offset map
    accs = sorted(sequences.keys())
    offsets = []
    cur = 0
    for acc in accs:
        L = len(sequences[acc])
        offsets.append((acc, cur, cur + L))
        cur += L

    # Build functional site lookup
    func_sites = {}
    for acc in accs:
        m = metadata.get(acc, {})
        sites = set()
        for feat in m.get("features", []):
            if feat.get("type") in ("Active site", "Binding site", "Disulfide bond"):
                for p in range(feat["start"] - 1, feat.get("end", feat["start"])):
                    sites.add(p)
        if sites:
            func_sites[acc] = sites

    with h5py.File(h5_path, "r") as f:
        total = f["activations"].shape[0]
        d_model = f["activations"].shape[1]
        holdout_start = int(total * (1 - holdout_frac))

        log.info(f"  Total residues: {total}, holdout: {total - holdout_start}, d_model: {d_model}")

        # Overall metrics
        total_mse = 0.0
        total_se_sum = 0.0
        total_x_sum = np.zeros(d_model, dtype=np.float64)
        total_x_sq_sum = np.zeros(d_model, dtype=np.float64)
        total_cos = 0.0
        total_l0 = 0.0
        n_samples = 0
        per_residue_errors = []

        # L0 distribution
        l0_values = []

        # Activation magnitude distribution
        act_magnitudes = []

        for start in range(holdout_start, total, batch_size):
            end = min(start + batch_size, total)
            x = torch.tensor(f["activations"][start:end], dtype=torch.float32)

            with torch.no_grad():
                z = sae.encode(x)
                x_hat = sae.decode(z)

            # Per-residue MSE
            per_res_mse = ((x - x_hat) ** 2).mean(dim=-1).numpy()
            per_residue_errors.extend(per_res_mse.tolist())

            # Accumulate
            total_se_sum += float(((x - x_hat) ** 2).sum())
            x_np = x.numpy()
            total_x_sum += x_np.sum(axis=0)
            total_x_sq_sum += (x_np ** 2).sum(axis=0)

            cos = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1)
            total_cos += float(cos.sum())

            l0 = (z > 0).float().sum(dim=-1)
            total_l0 += float(l0.sum())
            l0_values.extend(l0.numpy().tolist())

            # Activation magnitudes (sample)
            z_np = z.numpy()
            for row in z_np[:min(100, len(z_np))]:
                active = row[row > 0]
                if len(active) > 0:
                    act_magnitudes.extend(active.tolist()[:10])

            n_samples += (end - start)

        per_residue_errors = np.array(per_residue_errors)

        # Global metrics
        mean_x = total_x_sum / n_samples
        var_x = total_x_sq_sum / n_samples - mean_x ** 2
        total_var = float(var_x.sum())
        mean_mse = total_se_sum / (n_samples * d_model)
        pct_var = 1.0 - (total_se_sum / n_samples) / (total_var + 1e-10)

        overall = {
            "n_holdout_residues": n_samples,
            "d_model": d_model,
            "mean_recon_mse": float(mean_mse),
            "pct_variance_explained": float(pct_var),
            "mean_cosine_similarity": float(total_cos / n_samples),
            "mean_l0": float(total_l0 / n_samples),
            "median_l0": float(np.median(l0_values)),
            "std_l0": float(np.std(l0_values)),
            "l0_percentiles": {
                f"p{p}": float(np.percentile(l0_values, p))
                for p in [5, 25, 50, 75, 95]
            },
            "per_residue_mse_percentiles": {
                f"p{p}": float(np.percentile(per_residue_errors, p))
                for p in [5, 25, 50, 75, 95, 99]
            },
        }

        if act_magnitudes:
            overall["activation_magnitude_percentiles"] = {
                f"p{p}": float(np.percentile(act_magnitudes, p))
                for p in [25, 50, 75, 90, 95, 99]
            }

        # Analyze worst-reconstructed residues
        n_worst = min(2000, len(per_residue_errors))
        worst_idx = np.argsort(-per_residue_errors)[:n_worst]
        best_idx = np.argsort(per_residue_errors)[:n_worst]

        worst_analysis = {"worst": {}, "best": {}}
        for label, selected_idx in [("worst", worst_idx), ("best", best_idx)]:
            global_indices = holdout_start + selected_idx
            # Sort for linear scan through offsets
            sorted_gi = np.sort(global_indices)
            aa_counts = np.zeros(20)
            ss_counts = np.zeros(3)
            func_count = 0
            asa_vals = []

            oi = 0
            for gi in sorted_gi:
                while oi < len(offsets) - 1 and gi >= offsets[oi + 1][1]:
                    oi += 1
                acc, acc_start, acc_end = offsets[oi]
                pos = int(gi - acc_start)
                if pos < 0 or pos >= acc_end - acc_start:
                    continue
                seq = sequences.get(acc, "")
                if pos < len(seq) and seq[pos] in aa_to_idx:
                    aa_counts[aa_to_idx[seq[pos]]] += 1
                dssp = dssp_data.get(acc, [])
                if dssp and pos < len(dssp):
                    ss3 = dssp[pos].get("ss3", "C")
                    ss_counts[{"H": 0, "E": 1}.get(ss3, 2)] += 1
                    asa = dssp[pos].get("asa", 0)
                    if asa is not None:
                        asa_vals.append(float(asa))
                if acc in func_sites and pos in func_sites[acc]:
                    func_count += 1

            aa_bg = np.array([1/20]*20)  # uniform for simplicity
            aa_dist = aa_counts / (aa_counts.sum() + 1e-10)
            aa_enr = aa_dist / aa_bg
            top_aa_idx = int(np.argmax(aa_enr))

            worst_analysis[label] = {
                "n": int(len(selected_idx)),
                "mean_mse": float(per_residue_errors[selected_idx].mean()),
                "top_aa": AA_LIST[top_aa_idx],
                "top_aa_enrichment": float(aa_enr[top_aa_idx]),
                "ss_distribution": {
                    "H": float(ss_counts[0] / (ss_counts.sum() + 1e-10)),
                    "E": float(ss_counts[1] / (ss_counts.sum() + 1e-10)),
                    "C": float(ss_counts[2] / (ss_counts.sum() + 1e-10)),
                },
                "functional_frac": float(func_count / max(len(selected_idx), 1)),
                "mean_asa": float(np.mean(asa_vals)) if asa_vals else -1,
            }

        overall["residue_analysis"] = worst_analysis
        return overall


def main():
    out_dir = ROOT / "results" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json") as f:
        dssp_data = json.load(f)

    saes = [
        ("ESM-2 L24", "models/sae/esm2_scaled/layer_24_topk/best.pt",
         "data/activations/esm2_scaled/layer_24.h5"),
        ("ESM-3 S L33", "models/sae/esm3/S_layer_33_topk/best.pt",
         "data/activations/esm3_scaled_multimodal/S_layer_33.h5"),
        ("ESM-3 S+St L33", "models/sae/esm3/S_St_layer_33_topk/best.pt",
         "data/activations/esm3_scaled_multimodal/S+St_layer_33.h5"),
    ]

    all_results = {}
    for name, sae_path, h5_path in saes:
        log.info(f"\n{'='*50}")
        log.info(f"Assessing: {name}")
        log.info(f"{'='*50}")

        sae = load_sae(str(ROOT / sae_path))
        result = assess_reconstruction(
            sae, str(ROOT / h5_path), sequences, dssp_data, metadata)
        all_results[name] = result

        log.info(f"  MSE: {result['mean_recon_mse']:.6f}")
        log.info(f"  Variance explained: {result['pct_variance_explained']:.4f}")
        log.info(f"  Cosine similarity: {result['mean_cosine_similarity']:.4f}")
        log.info(f"  Mean L0: {result['mean_l0']:.1f}")
        log.info(f"  Worst residues: top AA={result['residue_analysis']['worst']['top_aa']} "
                 f"({result['residue_analysis']['worst']['top_aa_enrichment']:.1f}x)")

    # Summary
    log.info(f"\n{'='*70}")
    log.info(f"{'SAE':<18} {'MSE':>10} {'Var%':>8} {'CosSim':>8} {'L0':>6}")
    log.info("-" * 70)
    for name, r in all_results.items():
        log.info(f"{name:<18} {r['mean_recon_mse']:>10.6f} "
                 f"{r['pct_variance_explained']:>7.4f} "
                 f"{r['mean_cosine_similarity']:>7.4f} {r['mean_l0']:>6.1f}")

    out_path = out_dir / "sae_reconstruction_quality.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
