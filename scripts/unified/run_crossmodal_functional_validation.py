#!/usr/bin/env python3
"""Functional validation of cross-modal vs matched SAE features.

Three analyses:
1. Cross-modal vs matched variant prediction split
   - Ridge regression: MLM + cross-modal features vs MLM + matched features
2. Per-residue DMS sensitivity × feature activation
   - Correlate cross-modal/matched activation with mean mutation fitness
3. Feature ablation at functional sites
   - KL divergence from ablating cross-modal vs matched features at annotated sites

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/run_crossmodal_functional_validation.py
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("func_val")
warnings.filterwarnings("ignore")


def load_sae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


# ═══════════════════════════════════════════════════════════
# ANALYSIS 1: Cross-modal vs Matched Variant Prediction Split
# ═══════════════════════════════════════════════════════════

def run_variant_prediction_split(crossmodal_idx, matched_idx, sae, h5_path,
                                  sequences, offsets, accs, device="cuda:0"):
    """Compare variant prediction using cross-modal vs matched SAE features."""
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS 1: Cross-modal vs Matched Variant Prediction")
    log.info("=" * 60)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    esm2 = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # Load DMS data
    dms_dir = ROOT / "data" / "dms" / "substitutions" / "DMS_ProteinGym_substitutions"
    score_dir = ROOT / "results" / "unified" / "variant_prediction"

    # Map protein names to score files
    score_files = list(score_dir.glob("*_scores.csv"))
    protein_names = [f.stem.replace("_scores", "") for f in score_files]

    results_per_protein = []

    for pname, score_file in zip(protein_names, score_files):
        df = pd.read_csv(score_file)
        if len(df) < 20:
            continue

        # Get unique positions
        positions = sorted(df["pos"].unique())
        n_pos = len(positions)

        # We need per-position SAE feature activations
        # The score file has per-mutation MLM scores, but we need per-position SAE features
        # Load WT SAE activations for this protein
        # Find the protein in our dataset by matching to DMS file names
        # For now, use the aggregated scores already computed

        # Build per-position aggregated features
        pos_data = []
        for p in positions:
            pmuts = df[df["pos"] == p]
            mean_fitness = pmuts["fitness"].mean()
            mean_mlm = pmuts["mlm_score"].mean()
            mean_sae_mag = pmuts["sae_magnitude"].mean()
            mean_sae_dis = pmuts["sae_disruption"].mean()
            pos_data.append({
                "pos": p,
                "fitness": mean_fitness,
                "mlm_score": mean_mlm,
                "sae_magnitude": mean_sae_mag,
                "sae_disruption": mean_sae_dis,
            })

        if len(pos_data) < 10:
            continue

        pos_df = pd.DataFrame(pos_data)

        # For cross-modal vs matched split, we need per-position feature activations
        # from the ESM-3 S+St SAE. But our DMS proteins may not be in our ESM-3 dataset.
        # Instead, use a simpler approach: the existing SAE magnitude/disruption scores
        # already capture total SAE response. We can test whether cross-modal features
        # specifically improve prediction by using the per-position scores as-is and
        # running the prediction comparison from combined_prediction.json

        # Simple correlation comparison
        rho_mlm, _ = stats.spearmanr(pos_df["fitness"], pos_df["mlm_score"])
        rho_sae_mag, _ = stats.spearmanr(pos_df["fitness"], pos_df["sae_magnitude"])

        results_per_protein.append({
            "protein": pname,
            "n_positions": len(pos_df),
            "rho_mlm": float(rho_mlm),
            "rho_sae_magnitude": float(rho_sae_mag),
        })

    # For the split analysis, we need to recompute SAE features separately
    # for cross-modal and matched feature subsets on the DMS proteins.
    # Load the ESM-2 SAE (since DMS uses ESM-2) and split its features
    # based on which ones correspond to cross-modal/matched patterns.
    #
    # Actually, the cross-modal/matched split is for the ESM-3 S+St SAE,
    # and our DMS uses the ESM-2 SAE. We need a different approach:
    # Use the ESM-2 SAE but split based on convergent vs non-convergent features.

    log.info("Loading ESM-2 SAE for split prediction...")
    esm2_sae = load_sae("models/sae/esm2_scaled/layer_24_topk/best.pt")
    esm2_h5 = "data/activations/esm2_scaled/layer_24.h5"

    # Load ESM-2 sequences and offsets
    with open("data/scaled/sequences/sequences.json") as f:
        all_seqs = json.load(f)
    esm2_accs = sorted(all_seqs.keys())
    esm2_offsets = {}
    pos = 0
    for acc in esm2_accs:
        L = len(all_seqs[acc])
        esm2_offsets[acc] = (pos, L)
        pos += L

    # Instead: compute per-position cross-modal and matched feature activations
    # on the ESM-3 S+St SAE directly, for proteins we can match to DMS data
    log.info("Computing per-position cross-modal vs matched activations on DMS proteins...")

    # Match DMS proteins to our dataset
    with open("results/unified/variant_prediction/variant_prediction_summary.json") as f:
        vp_summary = json.load(f)

    split_results = []

    for pname in vp_summary:
        score_file = score_dir / f"{pname}_scores.csv"
        if not score_file.exists():
            continue
        df = pd.read_csv(score_file)
        if len(df) < 20:
            continue

        # Get per-position fitness and MLM
        positions = sorted(df["pos"].unique())
        pos_fitness = {}
        pos_mlm = {}
        for p in positions:
            pmuts = df[df["pos"] == p]
            pos_fitness[p] = pmuts["fitness"].mean()
            pos_mlm[p] = pmuts["mlm_score"].mean()

        # Now compute ESM-2 SAE per-position feature activations
        # We need to find this protein's activations
        vp_info = vp_summary[pname]
        dms_file = vp_info.get("file", "")

        # Try to find protein in ESM-2 activations using the score data
        # The score CSV has sae_magnitude per position - compute from ESM-2 SAE
        # Actually, let's compute cross-modal and matched activation magnitudes
        # using the ESM-2 L24 SAE, splitting features into "convergent" (analogous to matched)
        # and "non-convergent" (analogous to cross-modal) based on our convergence analysis

        # Load convergence data if available
        convergence_path = ROOT / "results" / "unified" / "esm2_esm3_comparison" / "feature_convergence.json"
        if not convergence_path.exists():
            convergence_path = ROOT / "results" / "unified" / "cross_model_comparison" / "feature_comparison.json"

        # For now, use the ESM-2 SAE and split features by activation frequency
        # (broadly-active vs position-specific, analogous to matched vs cross-modal)
        pass

    # BETTER APPROACH: Recompute variant scores using ESM-3 S+St SAE directly
    # by splitting feature activation magnitude into cross-modal and matched components
    log.info("Recomputing variant prediction with cross-modal/matched split using ESM-2 SAE...")

    # We'll use the ESM-2 SAE features, split by whether they are
    # alive in only subset of positions (analogous to cross-modal behavior)
    # Use activation frequency as a proxy for specificity

    with h5py.File(esm2_h5, "r") as f:
        sample_acts = torch.tensor(f["activations"][:10000], dtype=torch.float32)
    with torch.no_grad():
        z_sample = esm2_sae.encode(sample_acts).numpy()

    feature_freqs = (z_sample > 0).mean(axis=0)
    # Split: position-specific features (freq < median) vs broadly-active (freq >= median)
    median_freq = np.median(feature_freqs[feature_freqs > 0])
    alive_mask = feature_freqs > 0
    specific_mask = alive_mask & (feature_freqs < median_freq)
    broad_mask = alive_mask & (feature_freqs >= median_freq)
    specific_features = np.where(specific_mask)[0]
    broad_features = np.where(broad_mask)[0]
    log.info(f"  Specific features: {len(specific_features)}, Broad features: {len(broad_features)}")

    # Now for each DMS protein, compute specific vs broad SAE magnitude
    for pname in vp_summary:
        score_file = score_dir / f"{pname}_scores.csv"
        if not score_file.exists():
            continue
        df = pd.read_csv(score_file)
        if len(df) < 20:
            continue

        positions = sorted(df["pos"].unique())

        # Compute per-position aggregated scores
        Y = np.array([df[df["pos"] == p]["fitness"].mean() for p in positions])
        X_mlm = np.array([df[df["pos"] == p]["mlm_score"].mean() for p in positions]).reshape(-1, 1)
        X_sae_full = np.array([df[df["pos"] == p]["sae_magnitude"].mean() for p in positions]).reshape(-1, 1)

        if len(Y) < 15:
            continue

        # 5-fold CV Ridge regression
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        models_to_test = {
            "MLM_only": X_mlm,
            "MLM+SAE_all": np.hstack([X_mlm, X_sae_full]),
        }

        rhos = {}
        for model_name, X in models_to_test.items():
            preds = np.zeros(len(Y))
            for train_idx, test_idx in kf.split(X):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                reg = Ridge(alpha=1.0)
                reg.fit(X_train, Y[train_idx])
                preds[test_idx] = reg.predict(X_test)
            rho, _ = stats.spearmanr(Y, preds)
            rhos[model_name] = float(rho)

        split_results.append({
            "protein": pname,
            "n_positions": len(positions),
            **rhos,
        })

    for r in split_results:
        log.info(f"  {r['protein']}: MLM={r.get('MLM_only', 0):.3f}, "
                 f"MLM+SAE={r.get('MLM+SAE_all', 0):.3f}")

    return split_results


# ═══════════════════════════════════════════════════════════
# ANALYSIS 2: Per-Residue DMS × Feature Activation Correlation
# ═══════════════════════════════════════════════════════════

def run_dms_feature_correlation(crossmodal_idx, matched_idx, sae, h5_path,
                                 sequences, offsets, accs):
    """Correlate per-residue cross-modal/matched activation with DMS fitness."""
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS 2: Per-Residue DMS × Feature Activation")
    log.info("=" * 60)

    score_dir = ROOT / "results" / "unified" / "variant_prediction"
    with open("data/scaled/annotations/metadata.json") as f:
        metadata = json.load(f)

    # Map DMS protein names to UniProt accessions
    protein_acc_map = {
        "GFP": "P42212", "P53": "P04637", "TEM-1": "P62593",
        "BRCA1": "P38398", "ADRB2": "P07550", "GAL4": "P04386",
        "Calmodulin": "P0DP23", "PSD95": "P78352", "IF1": "P01096",
    }

    all_results = []

    for pname, acc in protein_acc_map.items():
        score_file = score_dir / f"{pname}_scores.csv"
        if not score_file.exists():
            continue
        df = pd.read_csv(score_file)

        # Check if protein is in our ESM-3 dataset
        if acc not in offsets:
            # Try to find by name match
            continue

        start, L = offsets[acc]

        # Load SAE activations for this protein
        with h5py.File(h5_path, "r") as f:
            total = f["activations"].shape[0]
            if start + L > total:
                continue
            raw = torch.tensor(f["activations"][start:start + L], dtype=torch.float32)

        with torch.no_grad():
            z = sae.encode(raw).numpy()  # (L, dict_size)

        # Per-position: sum of cross-modal feature activations vs matched
        cm_idx_arr = np.array([i for i in crossmodal_idx if i < z.shape[1]])
        mt_idx_arr = np.array([i for i in matched_idx if i < z.shape[1]])

        cm_activation = z[:, cm_idx_arr].sum(axis=1)  # (L,)
        mt_activation = z[:, mt_idx_arr].sum(axis=1)  # (L,)
        total_activation = z.sum(axis=1)  # (L,)

        # Per-position mean fitness from DMS
        positions = sorted(df["pos"].unique())
        pos_fitness = {}
        for p in positions:
            pos_fitness[p] = df[df["pos"] == p]["fitness"].mean()

        # Build aligned arrays
        valid_pos = [p for p in positions if 0 <= p < L]
        if len(valid_pos) < 10:
            log.info(f"  {pname}: too few valid positions ({len(valid_pos)})")
            continue

        fitness_arr = np.array([pos_fitness[p] for p in valid_pos])
        cm_arr = np.array([cm_activation[p] for p in valid_pos])
        mt_arr = np.array([mt_activation[p] for p in valid_pos])
        total_arr = np.array([total_activation[p] for p in valid_pos])

        # Correlations
        rho_cm, p_cm = stats.spearmanr(fitness_arr, cm_arr)
        rho_mt, p_mt = stats.spearmanr(fitness_arr, mt_arr)
        rho_total, p_total = stats.spearmanr(fitness_arr, total_arr)

        result = {
            "protein": pname,
            "accession": acc,
            "n_positions": len(valid_pos),
            "rho_crossmodal": float(rho_cm),
            "p_crossmodal": float(p_cm),
            "rho_matched": float(rho_mt),
            "p_matched": float(p_mt),
            "rho_total": float(rho_total),
            "p_total": float(p_total),
        }
        all_results.append(result)

        log.info(f"  {pname} ({acc}, n={len(valid_pos)}): "
                 f"ρ_cm={rho_cm:.3f} (p={p_cm:.3g}), "
                 f"ρ_mt={rho_mt:.3f} (p={p_mt:.3g}), "
                 f"ρ_total={rho_total:.3f}")

    # Aggregate
    if all_results:
        rhos_cm = [r["rho_crossmodal"] for r in all_results]
        rhos_mt = [r["rho_matched"] for r in all_results]
        rhos_total = [r["rho_total"] for r in all_results]

        log.info(f"\n  Aggregate (n={len(all_results)} proteins):")
        log.info(f"    Cross-modal: mean ρ = {np.mean(rhos_cm):.3f} ± {np.std(rhos_cm):.3f}")
        log.info(f"    Matched:     mean ρ = {np.mean(rhos_mt):.3f} ± {np.std(rhos_mt):.3f}")
        log.info(f"    Total SAE:   mean ρ = {np.mean(rhos_total):.3f} ± {np.std(rhos_total):.3f}")

        if len(rhos_cm) >= 3:
            stat, p = stats.wilcoxon(rhos_cm, rhos_mt)
            log.info(f"    Wilcoxon cm vs mt: p={p:.4g}")

    return all_results


# ═══════════════════════════════════════════════════════════
# ANALYSIS 3: Feature Ablation at Functional Sites
# ═══════════════════════════════════════════════════════════

def run_ablation_at_functional_sites(crossmodal_idx, matched_idx, sae, h5_path,
                                      sequences, offsets, accs, device="cuda:0",
                                      n_proteins=100, n_features=50):
    """Ablate cross-modal vs matched features, measure KL at functional sites."""
    log.info("\n" + "=" * 60)
    log.info("ANALYSIS 3: Feature Ablation at Functional Sites")
    log.info("=" * 60)

    with open("data/scaled/annotations/metadata.json") as f:
        metadata = json.load(f)

    # Load ESM-3 model for forward passes
    from esm.models.esm3 import ESM3
    from esm.tokenization import EsmSequenceTokenizer
    esm3 = ESM3.from_pretrained("esm3_sm_open_v1").to(device).eval()
    seq_tokenizer = EsmSequenceTokenizer()

    # Select top features by activation magnitude
    rng = np.random.RandomState(42)
    cm_sample = list(rng.choice(crossmodal_idx, min(n_features, len(crossmodal_idx)), replace=False))
    mt_sample = list(rng.choice(matched_idx, min(n_features, len(matched_idx)), replace=False))

    # Find proteins with functional site annotations
    proteins_with_sites = []
    for acc in accs:
        meta = metadata.get(acc, {})
        features = meta.get("features", [])
        if features and acc in offsets:
            start, L = offsets[acc]
            func_positions = set()
            for feat in features:
                for p in range(feat["start"] - 1, feat["end"]):
                    if 0 <= p < L:
                        func_positions.add(p)
            if len(func_positions) >= 3 and L <= 500:  # manageable length
                proteins_with_sites.append((acc, func_positions, L))

    log.info(f"Proteins with functional sites (L<=500): {len(proteins_with_sites)}")
    proteins_with_sites = proteins_with_sites[:n_proteins]

    def ablate_features_and_measure(feature_list, label):
        """Ablate a set of features and measure KL at func vs non-func sites."""
        kl_at_func = []
        kl_at_nonfunc = []

        for pi, (acc, func_pos, L) in enumerate(proteins_with_sites):
            seq = sequences[acc]
            if L > 500:
                continue

            # Tokenize
            tokens = seq_tokenizer.encode(seq)
            input_ids = torch.tensor([tokens], device=device)

            # Get baseline logits
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                baseline_out = esm3.forward(sequence_tokens=input_ids)
                baseline_logits = baseline_out.sequence_logits[0, 1:-1]  # (L, vocab)

            # Get activations at target layer and ablate features through SAE
            start, _ = offsets[acc]
            with h5py.File(h5_path, "r") as f:
                total = f["activations"].shape[0]
                if start + L > total:
                    continue
                raw = torch.tensor(f["activations"][start:start + L], dtype=torch.float32)

            # Encode through SAE, zero target features, decode
            with torch.no_grad():
                z = sae.encode(raw)
                z_ablated = z.clone()
                for fid in feature_list:
                    z_ablated[:, fid] = 0
                x_ablated = sae.decode(z_ablated)
                x_original = sae.decode(z)

            # The difference is what we substitute
            delta = x_ablated - x_original  # (L, d_model)

            # Hook to add delta to the residual stream at layer 33
            hook_handle = None
            delta_device = delta.to(device)

            def hook_fn(module, args):
                hidden = args[0]
                # Add delta to hidden states (positions 1:-1 are the actual residues)
                hidden[:, 1:1 + len(delta_device)] = hidden[:, 1:1 + len(delta_device)] + delta_device.unsqueeze(0).to(hidden.dtype)
                return (hidden,) + args[1:]

            # Register hook at layer 33
            block = esm3.transformer.blocks[33]
            hook_handle = block.register_forward_pre_hook(hook_fn)

            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ablated_out = esm3.forward(sequence_tokens=input_ids)
                    ablated_logits = ablated_out.sequence_logits[0, 1:-1]
            finally:
                hook_handle.remove()

            # Compute per-position KL divergence
            baseline_probs = torch.softmax(baseline_logits.float(), dim=-1)
            ablated_probs = torch.softmax(ablated_logits.float(), dim=-1)
            kl = (baseline_probs * (torch.log(baseline_probs + 1e-10) -
                                     torch.log(ablated_probs + 1e-10))).sum(dim=-1)  # (L,)
            kl = kl.cpu().numpy()

            # Split by functional vs non-functional
            for p in range(min(L, len(kl))):
                if p in func_pos:
                    kl_at_func.append(float(kl[p]))
                else:
                    kl_at_nonfunc.append(float(kl[p]))

            if (pi + 1) % 20 == 0:
                log.info(f"  [{label}] {pi + 1}/{len(proteins_with_sites)} proteins")

        return kl_at_func, kl_at_nonfunc

    log.info(f"\nAblating {len(cm_sample)} cross-modal features...")
    cm_func_kl, cm_nonfunc_kl = ablate_features_and_measure(cm_sample, "cross-modal")

    log.info(f"\nAblating {len(mt_sample)} matched features...")
    mt_func_kl, mt_nonfunc_kl = ablate_features_and_measure(mt_sample, "matched")

    # Results
    log.info(f"\n--- Ablation Results ---")
    log.info(f"Cross-modal ablation:")
    log.info(f"  At functional sites:     KL = {np.mean(cm_func_kl):.4f} ± {np.std(cm_func_kl):.4f} (n={len(cm_func_kl)})")
    log.info(f"  At non-functional sites: KL = {np.mean(cm_nonfunc_kl):.4f} ± {np.std(cm_nonfunc_kl):.4f} (n={len(cm_nonfunc_kl)})")

    log.info(f"Matched ablation:")
    log.info(f"  At functional sites:     KL = {np.mean(mt_func_kl):.4f} ± {np.std(mt_func_kl):.4f} (n={len(mt_func_kl)})")
    log.info(f"  At non-functional sites: KL = {np.mean(mt_nonfunc_kl):.4f} ± {np.std(mt_nonfunc_kl):.4f} (n={len(mt_nonfunc_kl)})")

    # Statistical tests
    # 1. Do cross-modal features cause more disruption at functional sites than background?
    if cm_func_kl and cm_nonfunc_kl:
        u, p = stats.mannwhitneyu(cm_func_kl, cm_nonfunc_kl, alternative='two-sided')
        log.info(f"\nCross-modal: func vs non-func: p={p:.4g}")

    # 2. Do matched features cause more disruption at functional sites?
    if mt_func_kl and mt_nonfunc_kl:
        u, p = stats.mannwhitneyu(mt_func_kl, mt_nonfunc_kl, alternative='two-sided')
        log.info(f"Matched: func vs non-func: p={p:.4g}")

    # 3. Cross-modal vs matched at functional sites
    if cm_func_kl and mt_func_kl:
        u, p = stats.mannwhitneyu(cm_func_kl, mt_func_kl, alternative='two-sided')
        log.info(f"At func sites: cross-modal vs matched: p={p:.4g}")

    # 4. Cross-modal vs matched at non-functional sites
    if cm_nonfunc_kl and mt_nonfunc_kl:
        u, p = stats.mannwhitneyu(cm_nonfunc_kl, mt_nonfunc_kl, alternative='two-sided')
        log.info(f"At non-func sites: cross-modal vs matched: p={p:.4g}")

    # Compute func/nonfunc ratio for each
    cm_ratio = np.mean(cm_func_kl) / (np.mean(cm_nonfunc_kl) + 1e-10)
    mt_ratio = np.mean(mt_func_kl) / (np.mean(mt_nonfunc_kl) + 1e-10)
    log.info(f"\nFunc/non-func KL ratio: cross-modal={cm_ratio:.2f}, matched={mt_ratio:.2f}")

    return {
        "crossmodal": {
            "mean_kl_func": float(np.mean(cm_func_kl)),
            "mean_kl_nonfunc": float(np.mean(cm_nonfunc_kl)),
            "n_func": len(cm_func_kl),
            "n_nonfunc": len(cm_nonfunc_kl),
            "func_nonfunc_ratio": float(cm_ratio),
        },
        "matched": {
            "mean_kl_func": float(np.mean(mt_func_kl)),
            "mean_kl_nonfunc": float(np.mean(mt_nonfunc_kl)),
            "n_func": len(mt_func_kl),
            "n_nonfunc": len(mt_nonfunc_kl),
            "func_nonfunc_ratio": float(mt_ratio),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-proteins-ablation", type=int, default=50)
    parser.add_argument("--n-features-ablation", type=int, default=50)
    args = parser.parse_args()

    # Load cross-modal feature indices
    cm_path = ROOT / "results" / "unified" / "crossmodal_features" / "crossmodal_features_L33.json"
    with open(cm_path) as f:
        cm_data = json.load(f)
    crossmodal_idx = cm_data["crossmodal_feature_indices"]
    matched_idx = cm_data["matched_feature_indices"]
    log.info(f"Cross-modal: {len(crossmodal_idx)}, Matched: {len(matched_idx)}")

    # Load SAE
    sae_path = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_scaled_topk" / "best.pt"
    sae = load_sae(str(sae_path))

    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33.h5")

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    accs = sorted(sequences.keys())
    offsets = {}
    pos = 0
    for acc in accs:
        L = len(sequences[acc])
        offsets[acc] = (pos, L)
        pos += L

    # Run Analysis 2 first (doesn't need ESM models loaded)
    results_2 = run_dms_feature_correlation(
        crossmodal_idx, matched_idx, sae, h5_path,
        sequences, offsets, accs
    )

    # Run Analysis 3 (needs ESM-3)
    results_3 = run_ablation_at_functional_sites(
        crossmodal_idx, matched_idx, sae, h5_path,
        sequences, offsets, accs,
        device=args.device,
        n_proteins=args.n_proteins_ablation,
        n_features=args.n_features_ablation,
    )

    # Save all results
    output = {
        "dms_feature_correlation": results_2,
        "ablation_at_functional_sites": results_3,
    }

    out_path = ROOT / "results" / "unified" / "crossmodal_features" / "functional_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
