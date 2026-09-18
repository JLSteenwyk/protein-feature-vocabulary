#!/usr/bin/env python3
"""DMS (Deep Mutational Scanning) correlation analysis for ESM-2 and ESM-3.

Correlates model-derived per-residue importance scores with experimental
mutational fitness effects from ProteinGym substitution datasets.

Model importance metrics per residue:
  1. Mutation sensitivity KL (mutate to all 19 AAs, measure mean KL)
  2. SAE feature activation magnitude (total activation at SAE layer)
  3. MLM prediction entropy (mask position, measure output entropy)

DMS metric per residue:
  - Mean absolute fitness effect across all mutations at that position

Run as:
    ./env/bin/python scripts/unified/run_dms_correlation.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_dms_correlation.py --model esm3 --device cuda:1

Output: results/unified/{model}/dms_correlation.json
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
from scipy import stats as scipy_stats


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "dms_correlation_log.txt", mode='w'),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("dms"), out_dir


# ProteinGym substitution datasets with well-characterized single mutants
# Local files in data/dms/substitutions/DMS_ProteinGym_substitutions/
DMS_DATASETS = [
    {"name": "GFP_Sarkisyan", "filename": "GFP_AEQVI_Sarkisyan_2016.csv"},
    {"name": "TEM1_Stiffler", "filename": "BLAT_ECOLX_Stiffler_2015.csv"},
    {"name": "GAL4_Kitzman", "filename": "GAL4_YEAST_Kitzman_2015.csv"},
    {"name": "UBE4B_Starita", "filename": "UBE4B_MOUSE_Starita_2013.csv"},
    {"name": "PTEN_Mighell", "filename": "PTEN_HUMAN_Mighell_2018.csv"},
    {"name": "BRCA1_Findlay", "filename": "BRCA1_HUMAN_Findlay_2018.csv"},
    {"name": "SPIKE_Starr_bind", "filename": "SPIKE_SARS2_Starr_2020_binding.csv"},
    {"name": "P53_Giacomelli", "filename": "P53_HUMAN_Giacomelli_2018_Null_Etoposide.csv"},
]


def load_dms_data(log):
    """Load ProteinGym DMS datasets from local files."""
    import pandas as pd

    dms_dir = ROOT / "data" / "dms" / "substitutions" / "DMS_ProteinGym_substitutions"

    datasets = []
    for ds in DMS_DATASETS:
        csv_path = dms_dir / ds["filename"]

        if not csv_path.exists():
            log.warning(f"  {ds['name']}: file not found at {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        if "mutant" not in df.columns:
            log.warning(f"  {ds['name']}: no 'mutant' column, skipping")
            continue

        # Filter to single-point mutations only
        single = df[~df["mutant"].str.contains(":")]
        if len(single) < 50:
            log.warning(f"  {ds['name']}: only {len(single)} single mutants, skipping")
            continue

        # Extract WT sequence from mutated_sequence by reverting mutation
        first_row = single.iloc[0]
        mut_code = first_row["mutant"]
        wt_aa_0 = mut_code[0]
        pos_0 = int(mut_code[1:-1]) - 1
        mut_seq = first_row["mutated_sequence"]
        wt_sequence = list(mut_seq)
        wt_sequence[pos_0] = wt_aa_0

        # Verify by reverting a few more mutations
        for _, row in single.head(20).iterrows():
            mc = row["mutant"]
            wt_aa = mc[0]
            pos = int(mc[1:-1]) - 1
            ms = list(row["mutated_sequence"])
            ms[pos] = wt_aa
            if "".join(ms) != "".join(wt_sequence):
                wt_sequence = ms
                break

        wt_sequence = "".join(wt_sequence)

        # Compute per-residue DMS effect
        median_score = single["DMS_score"].median()
        per_residue = {}
        for _, row in single.iterrows():
            mc = row["mutant"]
            pos = int(mc[1:-1]) - 1
            effect = abs(row["DMS_score"] - median_score)
            if pos not in per_residue:
                per_residue[pos] = []
            per_residue[pos].append(effect)

        per_residue = {pos: float(np.mean(vals)) for pos, vals in per_residue.items()}

        datasets.append({
            "name": ds["name"],
            "wt_sequence": wt_sequence,
            "per_residue_effect": per_residue,
            "n_variants": len(single),
            "n_positions": len(per_residue),
        })
        log.info(f"  {ds['name']}: {len(single)} variants, "
                 f"{len(per_residue)} positions, seq_len={len(wt_sequence)}")

    return datasets


STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def load_sae(model_name, layer, condition="S"):
    """Load a trained SAE checkpoint."""
    from sae.model import build_sae
    sae_dir = ROOT / "models" / "sae"
    cond = condition.replace("+", "_")

    if model_name == "esm2":
        for prefix in ["esm2_scaled", "esm2"]:
            for fname in ["best.pt", "final.pt"]:
                candidate = sae_dir / prefix / f"layer_{layer}_topk" / fname
                if candidate.exists():
                    ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
                    sae = build_sae(ckpt["sae_config"])
                    sae.load_state_dict(ckpt["model_state_dict"])
                    sae.eval()
                    return sae
    else:
        for prefix in ["esm3_scaled", "esm3"]:
            for fname in ["best.pt", "final.pt"]:
                candidate = sae_dir / prefix / f"{cond}_layer_{layer}_topk" / fname
                if candidate.exists():
                    ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
                    sae = build_sae(ckpt["sae_config"])
                    sae.load_state_dict(ckpt["model_state_dict"])
                    sae.eval()
                    return sae
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-positions", type=int, default=0,
                        help="Max DMS positions per protein for mutation KL (0=all)")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== DMS Correlation: {args.model} ===")
    device = args.device

    # Step 1: Load DMS data from local files
    log.info("Step 1: Loading DMS datasets...")
    datasets = load_dms_data(log)
    if not datasets:
        log.error("No DMS datasets available, exiting")
        return
    log.info(f"Loaded {len(datasets)} DMS datasets")

    # Step 2: Load model
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2, ESM2HookManager
        model, tokenizer = load_esm2(device=device)
        sae_layer = 24
    else:
        from models.esm3_hooks import load_esm3, ESM3HookManager
        model, tokenizers = load_esm3(device=device)
        sae_layer = 33

    # Load SAE
    sae = load_sae(args.model, sae_layer, "S")
    if sae is not None:
        sae = sae.to(device)
        log.info(f"Loaded SAE at layer {sae_layer}")

    # Setup hook manager
    if args.model == "esm2":
        hook_mgr = ESM2HookManager(model, layers=[sae_layer], extract_attention=False)
    else:
        hook_mgr = ESM3HookManager(model, layers=[sae_layer], extract_attention=False)

    from models.metrics import kl_divergence

    # Step 3: Process each DMS dataset
    per_protein = {}
    all_rhos = {"mutation_kl": [], "sae_activation": [], "mlm_entropy": []}

    for ds in datasets:
        name = ds["name"]
        seq = ds["wt_sequence"]
        per_res_effect = ds["per_residue_effect"]
        positions = sorted(per_res_effect.keys())
        L = len(seq)

        log.info(f"\n=== {name} (L={L}, {len(positions)} DMS positions) ===")

        # Skip if sequence has too many unknowns
        if seq.count('X') / max(len(seq), 1) > 0.3:
            log.warning(f"  Too many unknown residues, skipping")
            continue

        # Truncate if too long
        if L > 1022:
            log.info(f"  Truncating from {L} to 1022")
            L = 1022
            seq = seq[:L]
            positions = [p for p in positions if p < L]
            per_res_effect = {p: v for p, v in per_res_effect.items() if p < L}

        if len(positions) < 20:
            log.warning(f"  Too few positions ({len(positions)}), skipping")
            continue

        # Subsample positions if requested (for speed on large proteins)
        if args.max_positions > 0 and len(positions) > args.max_positions:
            rng = np.random.RandomState(42)
            positions = sorted(rng.choice(positions, args.max_positions, replace=False).tolist())
            per_res_effect = {p: per_res_effect[p] for p in positions}
            log.info(f"  Subsampled to {len(positions)} positions")

        t0 = time.time()

        try:
            # ---- Metric 1: Mutation KL per position ----
            log.info(f"  Computing mutation sensitivity KL...")
            mutation_kl = {}
            if args.model == "esm2":
                inputs = tokenizer(seq, return_tensors="pt",
                                   truncation=True, max_length=1024).to(device)
                with torch.no_grad():
                    wt_logits = model(**inputs).logits.cpu()

                for i, pos in enumerate(positions):
                    if pos >= L:
                        continue
                    wt_aa = seq[pos]
                    if wt_aa not in STANDARD_AAS:
                        continue
                    kls = []
                    for mut_aa in STANDARD_AAS:
                        if mut_aa == wt_aa:
                            continue
                        mut_seq = seq[:pos] + mut_aa + seq[pos+1:]
                        mut_inputs = tokenizer(mut_seq, return_tensors="pt",
                                               truncation=True, max_length=1024).to(device)
                        with torch.no_grad():
                            mut_logits = model(**mut_inputs).logits.cpu()
                        kl = kl_divergence(wt_logits, mut_logits).mean().item()
                        kls.append(kl)
                    mutation_kl[pos] = float(np.mean(kls))

                    if (i + 1) % 50 == 0:
                        log.info(f"    [{i+1}/{len(positions)} positions, "
                                 f"{time.time()-t0:.0f}s]")
            else:
                seq_tokens = tokenizers.sequence.encode(seq)
                seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                          device=device).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    wt_out = model(sequence_tokens=seq_tensor)
                wt_logits = (wt_out.sequence_logits if hasattr(wt_out, 'sequence_logits')
                             else wt_out.logits).float().cpu()

                for i, pos in enumerate(positions):
                    if pos >= L:
                        continue
                    wt_aa = seq[pos]
                    if wt_aa not in STANDARD_AAS:
                        continue
                    kls = []
                    for mut_aa in STANDARD_AAS:
                        if mut_aa == wt_aa:
                            continue
                        mut_seq = seq[:pos] + mut_aa + seq[pos+1:]
                        mut_tokens = tokenizers.sequence.encode(mut_seq)
                        mut_tensor = torch.tensor(mut_tokens, dtype=torch.long,
                                                  device=device).unsqueeze(0)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            mut_out = model(sequence_tokens=mut_tensor)
                        mut_logits = (mut_out.sequence_logits
                                      if hasattr(mut_out, 'sequence_logits')
                                      else mut_out.logits).float().cpu()
                        kl = kl_divergence(wt_logits, mut_logits).mean().item()
                        kls.append(kl)
                    mutation_kl[pos] = float(np.mean(kls))

                    if (i + 1) % 50 == 0:
                        log.info(f"    [{i+1}/{len(positions)} positions, "
                                 f"{time.time()-t0:.0f}s]")

            log.info(f"  Mutation KL: {len(mutation_kl)} positions, {time.time()-t0:.0f}s")

            # ---- Metric 2: SAE activation magnitude ----
            sae_mag = np.zeros(L)
            if sae is not None:
                if args.model == "esm2":
                    inputs = tokenizer(seq, return_tensors="pt",
                                       truncation=True, max_length=1024).to(device)
                    hook_mgr.cache.clear()
                    hook_mgr.register()
                    with torch.no_grad():
                        model(**inputs)
                    hook_mgr.remove()
                else:
                    seq_tokens = tokenizers.sequence.encode(seq)
                    seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                              device=device).unsqueeze(0)
                    hook_mgr.cache.clear()
                    hook_mgr.register()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        model(sequence_tokens=seq_tensor)
                    hook_mgr.remove()

                h = hook_mgr.cache.residual_stream[sae_layer][0][1:-1].float()
                with torch.no_grad():
                    z = sae.encode(h[:L].to(device))
                sae_mag = z.abs().sum(dim=-1).cpu().numpy()

            # ---- Metric 3: MLM entropy ----
            log.info(f"  Computing MLM entropy...")
            mlm_entropy = np.zeros(L)
            if args.model == "esm2":
                mask_id = tokenizer.mask_token_id
                base_inputs = tokenizer(seq, return_tensors="pt",
                                        truncation=True, max_length=1024)
                base_ids = base_inputs["input_ids"][0]

                chunk_size = 32
                for start in range(0, L, chunk_size):
                    end = min(start + chunk_size, L)
                    batch = base_ids.unsqueeze(0).expand(end - start, -1).clone()
                    for j in range(end - start):
                        batch[j, start + j + 1] = mask_id
                    batch = batch.to(device)
                    with torch.no_grad():
                        out = model(input_ids=batch)
                    probs = torch.softmax(out.logits.float(), dim=-1).cpu()
                    for j in range(end - start):
                        p = probs[j, start + j + 1]
                        mlm_entropy[start + j] = -torch.sum(p * torch.log(p + 1e-10)).item()
            else:
                seq_tokens = tokenizers.sequence.encode(seq)
                seq_t = torch.tensor(seq_tokens, dtype=torch.long)
                mask_id = 32

                # Adaptive chunk size: smaller for longer sequences to avoid OOM
                chunk_size = max(1, min(4, 2048 // max(L, 1)))
                for start in range(0, L, chunk_size):
                    end = min(start + chunk_size, L)
                    batch = seq_t.unsqueeze(0).expand(end - start, -1).clone()
                    for j in range(end - start):
                        batch[j, start + j + 1] = mask_id
                    batch = batch.to(device)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(sequence_tokens=batch)
                    logits = (out.sequence_logits if hasattr(out, 'sequence_logits')
                              else out.logits).float().cpu()
                    probs = torch.softmax(logits, dim=-1)
                    for j in range(end - start):
                        p = probs[j, start + j + 1]
                        mlm_entropy[start + j] = -torch.sum(p * torch.log(p + 1e-10)).item()

            # ---- Compute correlations ----
            shared_pos = sorted(set(positions) & set(mutation_kl.keys()))
            if len(shared_pos) < 10:
                log.warning(f"  Only {len(shared_pos)} shared positions, skipping")
                continue

            dms_vals = np.array([per_res_effect[p] for p in shared_pos])
            kl_vals = np.array([mutation_kl[p] for p in shared_pos])
            sae_vals = np.array([sae_mag[p] for p in shared_pos])
            ent_vals = np.array([mlm_entropy[p] for p in shared_pos])

            correlations = {}
            for metric_name, metric_vals in [("mutation_kl", kl_vals),
                                              ("sae_activation", sae_vals),
                                              ("mlm_entropy", ent_vals)]:
                if np.std(metric_vals) > 0 and np.std(dms_vals) > 0:
                    rho, p = scipy_stats.spearmanr(dms_vals, metric_vals)
                    r, pr = scipy_stats.pearsonr(dms_vals, metric_vals)
                else:
                    rho, p, r, pr = 0, 1, 0, 1

                correlations[metric_name] = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "pearson_r": float(r),
                    "pearson_p": float(pr),
                }

                if not np.isnan(rho):
                    all_rhos[metric_name].append(rho)

                log.info(f"  {metric_name}: ρ={rho:.3f} (p={p:.2e}), r={r:.3f}")

            per_protein[name] = {
                "n_residues": L,
                "n_dms_positions": len(shared_pos),
                "n_dms_variants": ds["n_variants"],
                "correlations": correlations,
            }

            log.info(f"  Total time: {time.time()-t0:.0f}s")

        except Exception as e:
            log.warning(f"  {name}: FAILED — {e}")
            torch.cuda.empty_cache()
            continue

    # ================================================================
    # Aggregate
    # ================================================================
    log.info(f"\n=== Aggregate Results ({len(per_protein)} proteins) ===")

    aggregate = {}
    for metric in ["mutation_kl", "sae_activation", "mlm_entropy"]:
        vals = all_rhos[metric]
        if vals:
            arr = np.array(vals)
            aggregate[metric] = {
                "mean_spearman": float(np.mean(arr)),
                "std_spearman": float(np.std(arr)),
                "median_spearman": float(np.median(arr)),
                "n": len(arr),
            }
            log.info(f"  {metric}: mean ρ = {np.mean(arr):.3f} ± {np.std(arr):.3f}")

    output = {
        "model": args.model,
        "n_dms_proteins": len(per_protein),
        "sae_layer": sae_layer,
        "per_protein": per_protein,
        "aggregate": aggregate,
    }

    out_path = out_dir / "dms_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
