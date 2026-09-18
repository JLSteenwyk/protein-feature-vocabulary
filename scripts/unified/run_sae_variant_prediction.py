#!/usr/bin/env python3
"""SAE feature-based variant effect prediction vs. raw model scores.

For each DMS protein:
1. Run ESM-2 forward pass → get layer 24 activations + MLM logits
2. Encode through SAE → per-residue feature vectors
3. Compute per-position scores:
   - MLM log-likelihood ratio (baseline)
   - SAE feature magnitude (L1 norm of SAE activations)
   - SAE feature disruption (L2 distance between WT and mutant SAE vectors)
   - SAE functional feature score (activations of known functional features)
4. Correlate each score with experimental DMS fitness
5. Report Spearman ρ comparison

Usage:
    ./env/bin/python scripts/unified/run_sae_variant_prediction.py --device cuda:0
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import pandas as pd
from scipy import stats

from sae.model import TopKSAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("variant_pred")


# Curated DMS proteins with known structure and manageable size
DMS_PROTEINS = [
    ("GFP_AEQVI_Sarkisyan_2016.csv", "GFP"),
    ("P53_HUMAN_Giacomelli_2018_Null_Etoposide.csv", "P53"),
    ("PABP_YEAST_Fields_2013-doubles.csv", "PABP"),
    ("BLAT_ECOLX_Stiffler_2015.csv", "TEM-1"),
    ("BRCA1_HUMAN_Findlay_2018.csv", "BRCA1"),
    ("ADRB2_HUMAN_Jones_2020.csv", "ADRB2"),
    ("GAL4_YEAST_Kitzman_2015.csv", "GAL4"),
    ("UBE4B_MOUSE_Klevit_2013-singles.csv", "UBE4B"),
    ("CALM1_HUMAN_Weile_2017.csv", "Calmodulin"),
    ("HSP82_YEAST_Flynn_2015.csv", "HSP90"),
    ("DLG4_RAT_McLaughlin_2012.csv", "PSD95"),
    ("IF1_ECOLI_Kelsic_2016.csv", "IF1"),
]


def load_esm2(device):
    """Load ESM-2 650M model."""
    from transformers import AutoTokenizer, EsmForMaskedLM
    model_name = "facebook/esm2_t33_650M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()
    return model, tokenizer


def load_sae_esm2():
    """Load ESM-2 layer 24 SAE."""
    ckpt = torch.load("models/sae/esm2_scaled/layer_24_topk/best.pt",
                       map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def get_wt_sequence_from_dms(df):
    """Extract wild-type sequence from DMS dataframe."""
    row = df.iloc[0]
    mutant = row["mutant"]
    mut_seq = row["mutated_sequence"]

    # Handle single and multiple mutations
    mutations = mutant.split(":")
    wt_seq = list(mut_seq)
    for m in mutations:
        wt_aa = m[0]
        pos = int(m[1:-1]) - 1  # 0-indexed
        wt_seq[pos] = wt_aa
    return "".join(wt_seq)


def extract_layer_activations(model, tokenizer, sequence, layer=24, device="cpu"):
    """Run ESM-2 and extract activations at a specific layer + MLM logits."""
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Layer activations (skip CLS/EOS tokens)
    hidden = outputs.hidden_states[layer][0, 1:-1].cpu()  # (L, 1280)

    # MLM logits
    logits = outputs.logits[0, 1:-1].cpu()  # (L, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)

    return hidden, log_probs


def compute_sae_scores(sae, wt_hidden, mut_hidden_dict, positions):
    """Compute SAE-based variant effect scores.

    Args:
        sae: trained SAE
        wt_hidden: (L, 1280) WT activations
        mut_hidden_dict: {pos: (1280,) mutant activation at that position}
        positions: list of mutation positions

    Returns dict of score arrays aligned with positions.
    """
    with torch.no_grad():
        z_wt = sae.encode(wt_hidden)  # (L, dict_size)

    scores = {
        "sae_magnitude": [],      # L1 norm at WT position
        "sae_n_features": [],     # Number of active features at WT position
        "sae_disruption": [],     # L2(z_wt - z_mut) at mutation position
    }

    for pos in positions:
        # WT scores at this position
        z_pos = z_wt[pos]
        scores["sae_magnitude"].append(float(z_pos.abs().sum()))
        scores["sae_n_features"].append(int((z_pos > 0).sum()))

        # Disruption score (if mutant activations available)
        if pos in mut_hidden_dict:
            mut_h = mut_hidden_dict[pos].unsqueeze(0)
            with torch.no_grad():
                z_mut = sae.encode(mut_h)[0]
            disruption = float(torch.norm(z_pos - z_mut, p=2))
            scores["sae_disruption"].append(disruption)
        else:
            scores["sae_disruption"].append(np.nan)

    return {k: np.array(v) for k, v in scores.items()}


def compute_mlm_score(log_probs, tokenizer, wt_aa, mut_aa, pos):
    """Compute log-likelihood ratio for a single mutation."""
    # Get token IDs
    wt_tok = tokenizer.encode(wt_aa, add_special_tokens=False)
    mut_tok = tokenizer.encode(mut_aa, add_special_tokens=False)

    if not wt_tok or not mut_tok:
        return np.nan

    wt_logp = float(log_probs[pos, wt_tok[0]])
    mut_logp = float(log_probs[pos, mut_tok[0]])

    return mut_logp - wt_logp  # Positive = mutation is favored


def process_dms_protein(dms_path, model, tokenizer, sae, device,
                        max_mutations=2000):
    """Process one DMS protein and return all scores + fitness."""
    df = pd.read_csv(dms_path)

    # Filter to single-point mutations only
    singles = df[~df["mutant"].str.contains(":")]
    if len(singles) < 20:
        log.warning(f"  Too few single mutations: {len(singles)}")
        return None

    if len(singles) > max_mutations:
        singles = singles.sample(max_mutations, random_state=42)

    wt_seq = get_wt_sequence_from_dms(df)
    L = len(wt_seq)
    log.info(f"  WT length: {L}, mutations: {len(singles)}")

    if L > 1022:  # ESM-2 max length
        log.warning(f"  Sequence too long ({L}), truncating to 1022")
        wt_seq = wt_seq[:1022]
        L = 1022
        singles = singles[singles["mutant"].apply(
            lambda m: int(m[1:-1]) <= L
        )]
        if len(singles) < 20:
            return None

    # Get WT activations and logits
    wt_hidden, wt_log_probs = extract_layer_activations(
        model, tokenizer, wt_seq, layer=24, device=device
    )

    # For disruption: run mutants in batches grouped by position
    # For efficiency, only compute disruption for a subset of unique positions
    unique_positions = sorted(set(int(m[1:-1]) - 1 for m in singles["mutant"]))

    # Run mutant forward passes for disruption scores
    # Group mutations by position to avoid redundant runs
    mut_hidden_dict = {}
    positions_to_run = unique_positions[:200]  # Cap at 200 for speed

    log.info(f"  Computing disruption for {len(positions_to_run)} positions...")
    for i, pos in enumerate(positions_to_run):
        if pos >= L:
            continue
        # Get all mutations at this position
        pos_muts = singles[singles["mutant"].apply(lambda m: int(m[1:-1]) - 1 == pos)]
        if len(pos_muts) == 0:
            continue

        # Use first mutation at this position for the disruption score
        first_mut = pos_muts.iloc[0]["mutant"]
        mut_aa = first_mut[-1]
        mut_seq = list(wt_seq)
        mut_seq[pos] = mut_aa
        mut_seq_str = "".join(mut_seq)

        try:
            mut_hidden, _ = extract_layer_activations(
                model, tokenizer, mut_seq_str, layer=24, device=device
            )
            mut_hidden_dict[pos] = mut_hidden[pos]
        except Exception as e:
            log.warning(f"  Error at pos {pos}: {e}")
            continue

        if (i + 1) % 50 == 0:
            log.info(f"    {i+1}/{len(positions_to_run)} positions")

    # Compute all scores
    results = []
    for _, row in singles.iterrows():
        mutant = row["mutant"]
        wt_aa = mutant[0]
        mut_aa = mutant[-1]
        pos = int(mutant[1:-1]) - 1  # 0-indexed
        fitness = row["DMS_score"]

        if pos >= L:
            continue

        # MLM score
        mlm_score = compute_mlm_score(wt_log_probs, tokenizer, wt_aa, mut_aa, pos)

        # SAE scores at WT position
        with torch.no_grad():
            z_wt = sae.encode(wt_hidden)
        z_pos = z_wt[pos]
        sae_mag = float(z_pos.abs().sum())
        sae_nfeat = int((z_pos > 0).sum())

        # Disruption score
        if pos in mut_hidden_dict:
            with torch.no_grad():
                z_mut = sae.encode(mut_hidden_dict[pos].unsqueeze(0))[0]
            sae_disruption = float(torch.norm(z_pos - z_mut, p=2))
        else:
            sae_disruption = np.nan

        results.append({
            "mutant": mutant,
            "pos": pos,
            "fitness": fitness,
            "mlm_score": mlm_score,
            "sae_magnitude": sae_mag,
            "sae_n_features": sae_nfeat,
            "sae_disruption": sae_disruption,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=12)
    parser.add_argument("--max-mutations", type=int, default=2000)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "unified" / "variant_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    dms_dir = ROOT / "data" / "dms" / "substitutions" / "DMS_ProteinGym_substitutions"

    # Load models
    log.info("Loading ESM-2...")
    model, tokenizer = load_esm2(args.device)
    log.info("Loading SAE...")
    sae = load_sae_esm2()

    all_results = {}
    summary_rows = []

    for dms_file, name in DMS_PROTEINS[:args.max_proteins]:
        dms_path = dms_dir / dms_file
        if not dms_path.exists():
            log.warning(f"Missing: {dms_path}")
            continue

        log.info(f"\n=== {name} ({dms_file}) ===")
        try:
            df = process_dms_protein(
                str(dms_path), model, tokenizer, sae, args.device,
                max_mutations=args.max_mutations
            )
        except Exception as e:
            log.error(f"  Failed: {e}")
            torch.cuda.empty_cache()
            continue

        if df is None or len(df) < 20:
            log.warning(f"  Insufficient data")
            continue

        # Compute correlations
        scores = {}
        for col in ["mlm_score", "sae_magnitude", "sae_n_features", "sae_disruption"]:
            valid = df.dropna(subset=[col, "fitness"])
            if len(valid) < 20:
                scores[col] = {"rho": np.nan, "p": np.nan, "n": 0}
                continue
            rho, p = stats.spearmanr(valid[col], valid["fitness"])
            scores[col] = {"rho": float(rho), "p": float(p), "n": len(valid)}

        log.info(f"  Results ({len(df)} mutations):")
        for col, s in scores.items():
            log.info(f"    {col}: ρ={s['rho']:.3f} (p={s['p']:.3g}, n={s['n']})")

        # Does SAE disruption beat MLM?
        mlm_rho = abs(scores["mlm_score"]["rho"]) if not np.isnan(scores["mlm_score"]["rho"]) else 0
        dis_rho = abs(scores["sae_disruption"]["rho"]) if not np.isnan(scores["sae_disruption"]["rho"]) else 0
        mag_rho = abs(scores["sae_magnitude"]["rho"]) if not np.isnan(scores["sae_magnitude"]["rho"]) else 0

        improvement = max(dis_rho, mag_rho) - mlm_rho
        log.info(f"  Best SAE |ρ|={max(dis_rho, mag_rho):.3f} vs MLM |ρ|={mlm_rho:.3f} (Δ={improvement:+.3f})")

        all_results[name] = {
            "file": dms_file,
            "n_mutations": len(df),
            "scores": scores,
        }
        summary_rows.append({
            "protein": name,
            "n": len(df),
            "mlm_rho": scores["mlm_score"]["rho"],
            "sae_mag_rho": scores["sae_magnitude"]["rho"],
            "sae_disruption_rho": scores["sae_disruption"]["rho"],
            "sae_nfeat_rho": scores["sae_n_features"]["rho"],
        })

        # Save per-protein results
        df.to_csv(out_dir / f"{name}_scores.csv", index=False)

        torch.cuda.empty_cache()

    # Save summary
    with open(out_dir / "variant_prediction_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        log.info(f"\n{'='*70}")
        log.info("SUMMARY: Spearman ρ with DMS fitness")
        log.info(f"{'='*70}")
        log.info(f"{'Protein':<12} {'n':>5} {'MLM':>8} {'SAE mag':>8} {'SAE disr':>8} {'SAE nfeat':>9}")
        log.info(f"{'-'*12} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
        for _, row in summary_df.iterrows():
            log.info(f"{row['protein']:<12} {row['n']:>5} {row['mlm_rho']:>8.3f} "
                    f"{row['sae_mag_rho']:>8.3f} {row['sae_disruption_rho']:>8.3f} "
                    f"{row['sae_nfeat_rho']:>9.3f}")

        # Aggregate
        log.info(f"{'-'*12} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
        log.info(f"{'Mean':<12} {'':>5} {summary_df['mlm_rho'].mean():>8.3f} "
                f"{summary_df['sae_mag_rho'].mean():>8.3f} "
                f"{summary_df['sae_disruption_rho'].dropna().mean():>8.3f} "
                f"{summary_df['sae_nfeat_rho'].mean():>9.3f}")

        summary_df.to_csv(out_dir / "variant_prediction_comparison.csv", index=False)

    log.info("\nDone!")


if __name__ == "__main__":
    main()
