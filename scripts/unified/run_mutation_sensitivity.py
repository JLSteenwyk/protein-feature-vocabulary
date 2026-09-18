#!/usr/bin/env python3
"""Mutation sensitivity analysis for ESM-2 and ESM-3.

For each protein, introduce single-site mutations and measure how
model predictions change. Compares sensitivity at:
- Functional sites (active sites, binding sites) vs non-functional
- Buried vs exposed residues (from DSSP ASA)
- Secondary structure elements vs coil

This reveals which positions the model considers most important and
whether that aligns with known biology.

Run as:
    ./env/bin/python scripts/unified/run_mutation_sensitivity.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_mutation_sensitivity.py --model esm3 --device cuda:1

Output: results/unified/{model}/mutation_sensitivity.json
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

from models.metrics import kl_divergence


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "mutation_sensitivity_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("mutation"), out_dir


def load_sequences(max_proteins=500, max_length=500):
    seq_file = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    if seq_file.exists():
        with open(seq_file) as f:
            all_seqs = json.load(f)
    else:
        all_seqs = {}
        fasta_file = ROOT / "data" / "scaled" / "sequences" / "sequences.fasta"
        if fasta_file.exists():
            acc, seq = None, []
            with open(fasta_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if acc and seq:
                            all_seqs[acc] = "".join(seq)
                        acc = line[1:].split()[0]
                        seq = []
                    else:
                        seq.append(line)
            if acc and seq:
                all_seqs[acc] = "".join(seq)
    filtered = {k: v for k, v in all_seqs.items() if len(v) <= max_length}
    sorted_accs = sorted(filtered.keys())[:max_proteins]
    return {k: filtered[k] for k in sorted_accs}


def load_annotations():
    """Load DSSP and functional site annotations."""
    dssp_file = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"
    meta_file = ROOT / "data" / "scaled" / "annotations" / "metadata.json"

    dssp = {}
    if dssp_file.exists():
        with open(dssp_file) as f:
            dssp = json.load(f)

    functional_sites = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        for acc, data in meta.items():
            sites = set()
            for feat in data.get("features", []):
                if feat.get("type") in ("Active site", "Binding site", "Metal binding",
                                         "Site", "Disulfide bond"):
                    for pos in range(feat["start"], feat["end"] + 1):
                        sites.add(pos - 1)  # 0-indexed
            if sites:
                functional_sites[acc] = sites

    return dssp, functional_sites


STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def compute_wt_logits_esm2(model, tokenizer, seq, device):
    """Get wild-type logits from ESM-2."""
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model(**inputs)
    return out.logits.cpu()  # (1, L+2, V)


def compute_mutant_sensitivity_esm2(model, tokenizer, seq, positions, device):
    """For each position, mutate to all other AAs and measure KL divergence.

    Returns per-position mean KL divergence (sensitivity score).
    """
    wt_logits = compute_wt_logits_esm2(model, tokenizer, seq, device)
    sensitivities = {}

    for pos in positions:
        wt_aa = seq[pos]
        kl_scores = []
        for mut_aa in STANDARD_AAS:
            if mut_aa == wt_aa:
                continue
            mut_seq = seq[:pos] + mut_aa + seq[pos+1:]
            inputs = tokenizer(mut_seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with torch.no_grad():
                out = model(**inputs)
            mut_logits = out.logits.cpu()
            kl = kl_divergence(wt_logits, mut_logits)
            kl_scores.append(kl.mean().item())

        sensitivities[pos] = float(np.mean(kl_scores))
    return sensitivities


def compute_wt_logits_esm3(model, tokenizers, seq, device):
    """Get wild-type logits from ESM-3."""
    seq_tokens = tokenizers.sequence.encode(seq)
    seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(sequence_tokens=seq_tensor)
    logits = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
    return logits.float().cpu()


def compute_mutant_sensitivity_esm3(model, tokenizers, seq, positions, device):
    """For each position, mutate and measure KL divergence."""
    wt_logits = compute_wt_logits_esm3(model, tokenizers, seq, device)
    sensitivities = {}

    for pos in positions:
        wt_aa = seq[pos]
        kl_scores = []
        for mut_aa in STANDARD_AAS:
            if mut_aa == wt_aa:
                continue
            mut_seq = seq[:pos] + mut_aa + seq[pos+1:]
            seq_tokens = tokenizers.sequence.encode(mut_seq)
            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(sequence_tokens=seq_tensor)
            logits = (out.sequence_logits if hasattr(out, 'sequence_logits')
                      else out.logits).float().cpu()
            kl = kl_divergence(wt_logits, logits)
            kl_scores.append(kl.mean().item())

        sensitivities[pos] = float(np.mean(kl_scores))
    return sensitivities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--positions-per-protein", type=int, default=20,
                        help="Number of positions to mutate per protein (sampled)")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Mutation Sensitivity: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    dssp, functional_sites = load_annotations()
    log.info(f"Loaded {len(sequences)} proteins, "
             f"{len(dssp)} with DSSP, {len(functional_sites)} with functional sites")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)

    # Collect sensitivity scores by category
    func_sensitivities = []
    nonfunc_sensitivities = []
    buried_sensitivities = []
    exposed_sensitivities = []
    helix_sensitivities = []
    sheet_sensitivities = []
    coil_sensitivities = []

    t0 = time.time()
    n_evaluated = 0

    for seq_idx, (acc, seq) in enumerate(sequences.items()):
        L = len(seq)
        if L < 20:
            continue

        # Select positions to mutate
        # Prefer functional sites + random sample
        func_pos = sorted(functional_sites.get(acc, set()))
        func_pos = [p for p in func_pos if p < L]
        nonfunc_pool = [p for p in range(L) if p not in func_pos]

        # Take all functional + sample non-functional
        n_nonfunc = max(0, args.positions_per_protein - len(func_pos))
        if nonfunc_pool and n_nonfunc > 0:
            rng = np.random.RandomState(seq_idx)
            nonfunc_sample = sorted(rng.choice(nonfunc_pool,
                                                min(n_nonfunc, len(nonfunc_pool)),
                                                replace=False).tolist())
        else:
            nonfunc_sample = []

        all_positions = sorted(set(func_pos + nonfunc_sample))
        if not all_positions:
            continue

        # Compute sensitivities
        if args.model == "esm2":
            sens = compute_mutant_sensitivity_esm2(
                model, tokenizer, seq, all_positions, args.device)
        else:
            sens = compute_mutant_sensitivity_esm3(
                model, tokenizers, seq, all_positions, args.device)

        # Categorize
        dssp_data = dssp.get(acc, [])
        func_set = functional_sites.get(acc, set())

        for pos, kl in sens.items():
            if pos in func_set:
                func_sensitivities.append(kl)
            else:
                nonfunc_sensitivities.append(kl)

            if pos < len(dssp_data):
                r = dssp_data[pos]
                asa = r.get('asa', 50)
                ss3 = r.get('ss3', 'C')
                if asa < 25:
                    buried_sensitivities.append(kl)
                else:
                    exposed_sensitivities.append(kl)
                if ss3 == 'H':
                    helix_sensitivities.append(kl)
                elif ss3 == 'E':
                    sheet_sensitivities.append(kl)
                else:
                    coil_sensitivities.append(kl)

        n_evaluated += 1
        if n_evaluated % 20 == 0:
            log.info(f"  [{n_evaluated} proteins, {time.time()-t0:.0f}s] "
                     f"func={np.mean(func_sensitivities) if func_sensitivities else 0:.4f} "
                     f"nonfunc={np.mean(nonfunc_sensitivities) if nonfunc_sensitivities else 0:.4f}")

    log.info(f"Evaluated {n_evaluated} proteins in {time.time()-t0:.1f}s")

    def summarize(values, name):
        if not values:
            return {"mean": 0, "std": 0, "n": 0}
        log.info(f"  {name}: mean={np.mean(values):.6f} +/- {np.std(values):.6f} (n={len(values)})")
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "n": len(values),
        }

    log.info("=== Summary ===")
    output = {
        "model": args.model,
        "n_proteins": n_evaluated,
        "positions_per_protein": args.positions_per_protein,
        "functional_vs_nonfunctional": {
            "functional": summarize(func_sensitivities, "Functional"),
            "nonfunctional": summarize(nonfunc_sensitivities, "Non-functional"),
            "ratio": (float(np.mean(func_sensitivities)) / (float(np.mean(nonfunc_sensitivities)) + 1e-10)
                      if func_sensitivities and nonfunc_sensitivities else 0),
        },
        "buried_vs_exposed": {
            "buried": summarize(buried_sensitivities, "Buried"),
            "exposed": summarize(exposed_sensitivities, "Exposed"),
            "ratio": (float(np.mean(buried_sensitivities)) / (float(np.mean(exposed_sensitivities)) + 1e-10)
                      if buried_sensitivities and exposed_sensitivities else 0),
        },
        "secondary_structure": {
            "helix": summarize(helix_sensitivities, "Helix"),
            "sheet": summarize(sheet_sensitivities, "Sheet"),
            "coil": summarize(coil_sensitivities, "Coil"),
        },
    }

    if func_sensitivities and nonfunc_sensitivities:
        # Mann-Whitney U test
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(func_sensitivities, nonfunc_sensitivities, alternative='greater')
        output["functional_vs_nonfunctional"]["mannwhitney_p"] = float(pval)
        log.info(f"Functional > Non-functional: U={stat:.0f}, p={pval:.2e}")

    out_path = out_dir / "mutation_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
