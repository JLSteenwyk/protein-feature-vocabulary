#!/usr/bin/env python3
"""Evolutionary conservation correlation analysis for ESM-2 and ESM-3.

Computes per-residue model importance metrics and correlates them with
evolutionary conservation (approximated via MLM pseudo-perplexity).

Model importance metrics:
  1. Mutation sensitivity KL (subsample of positions)
  2. SAE feature activation magnitude
  3. MLM prediction entropy (low entropy = model "knows" the position)

Conservation proxy:
  - MLM pseudo-conservation: mask each position and measure prediction entropy
    (high confidence = conserved, since the model learned evolutionary patterns)

We also validate against DSSP accessibility and functional site annotations.

Run as:
    ./env/bin/python scripts/unified/run_evolutionary_correlation.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_evolutionary_correlation.py --model esm3 --device cuda:1

Output: results/unified/{model}/evolutionary_correlation.json
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
            logging.FileHandler(out_dir / "evolutionary_correlation_log.txt", mode='w'),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("evolution"), out_dir


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


STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"


def compute_mlm_entropy_esm2(model, tokenizer, seq, device):
    """Compute per-position prediction entropy by masking each position.

    Uses batched inference: mask each position separately and run all in one batch.
    Returns array of shape (L,) with entropy per position.
    """
    L = len(seq)
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"][0]  # (L+2,)

    # Get mask token id
    mask_id = tokenizer.mask_token_id

    # Create batched masked inputs (L copies, each with one position masked)
    # Position 0 is BOS, position L+1 is EOS, residues are 1..L
    batch_ids = input_ids.unsqueeze(0).expand(L, -1).clone()
    for i in range(L):
        batch_ids[i, i + 1] = mask_id  # +1 for BOS

    # Process in chunks to avoid OOM
    chunk_size = 32
    entropies = np.zeros(L)
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        chunk = batch_ids[start:end].to(device)
        attn_mask = torch.ones_like(chunk).to(device)
        with torch.no_grad():
            out = model(input_ids=chunk, attention_mask=attn_mask)
        logits = out.logits.float().cpu()  # (chunk, L+2, V)
        probs = torch.softmax(logits, dim=-1)
        for i in range(end - start):
            pos = start + i
            p = probs[i, pos + 1]  # +1 for BOS
            ent = -torch.sum(p * torch.log(p + 1e-10)).item()
            entropies[pos] = ent

    return entropies


def compute_mlm_entropy_esm3(model, tokenizers, seq, device):
    """Compute per-position prediction entropy for ESM-3."""
    L = len(seq)
    seq_tokens = tokenizers.sequence.encode(seq)
    seq_tensor = torch.tensor(seq_tokens, dtype=torch.long)  # (L+2,)

    # Get mask token id (typically 32 for ESM-3)
    mask_id = 32  # ESM-3 mask token

    entropies = np.zeros(L)
    chunk_size = max(1, min(4, 2048 // max(L, 1)))  # adaptive: smaller for long seqs
    for start in range(0, L, chunk_size):
        end = min(start + chunk_size, L)
        batch = seq_tensor.unsqueeze(0).expand(end - start, -1).clone()
        for i in range(end - start):
            batch[i, start + i + 1] = mask_id  # +1 for BOS

        batch = batch.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(sequence_tokens=batch)
        logits = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
        logits = logits.float().cpu()
        probs = torch.softmax(logits, dim=-1)
        for i in range(end - start):
            pos = start + i
            p = probs[i, pos + 1]  # +1 for BOS
            ent = -torch.sum(p * torch.log(p + 1e-10)).item()
            entropies[pos] = ent

    return entropies


def compute_sae_activation_magnitude(sae, hidden_states, device):
    """Compute per-position total SAE activation magnitude."""
    # hidden_states: (L, d_model)
    with torch.no_grad():
        z = sae.encode(hidden_states.to(device))  # (L, dict_size)
    return z.abs().sum(dim=-1).cpu().numpy()  # (L,)


def compute_mutation_kl_subset(model, tokenizer_or_tokenizers, seq, positions,
                                model_name, device):
    """Compute mutation sensitivity KL for a subset of positions."""
    from models.metrics import kl_divergence

    if model_name == "esm2":
        tokenizer = tokenizer_or_tokenizers
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            wt_logits = model(**inputs).logits.cpu()

        sensitivities = {}
        for pos in positions:
            wt_aa = seq[pos]
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
            sensitivities[pos] = float(np.mean(kls))
        return sensitivities
    else:
        tokenizers = tokenizer_or_tokenizers
        seq_tokens = tokenizers.sequence.encode(seq)
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(sequence_tokens=seq_tensor)
        wt_logits = (out.sequence_logits if hasattr(out, 'sequence_logits')
                     else out.logits).float().cpu()

        sensitivities = {}
        for pos in positions:
            wt_aa = seq[pos]
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
                mut_logits = (mut_out.sequence_logits if hasattr(mut_out, 'sequence_logits')
                              else mut_out.logits).float().cpu()
                kl = kl_divergence(wt_logits, mut_logits).mean().item()
                kls.append(kl)
            sensitivities[pos] = float(np.mean(kls))
        return sensitivities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=400)
    parser.add_argument("--mutation-positions-per-protein", type=int, default=15)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Evolutionary Correlation: {args.model} ===")
    device = args.device

    # Load sequences
    seq_file = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    with open(seq_file) as f:
        all_seqs = json.load(f)
    seqs = {k: v for k, v in all_seqs.items() if len(v) <= args.max_length}
    sorted_accs = sorted(seqs.keys())[:args.max_proteins]
    seqs = {k: seqs[k] for k in sorted_accs}

    # Load annotations
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
                if feat.get("type") in ("Active site", "Binding site",
                                         "Metal binding", "Site"):
                    for pos in range(feat["start"], feat["end"] + 1):
                        sites.add(pos - 1)
            if sites:
                functional_sites[acc] = sites

    log.info(f"Loaded {len(seqs)} proteins, {len(dssp)} with DSSP, "
             f"{len(functional_sites)} with functional sites")

    # Load model
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2, ESM2HookManager
        model, tokenizer = load_esm2(device=device)
        sae_layer = 24
        tok_ref = tokenizer
    else:
        from models.esm3_hooks import load_esm3, ESM3HookManager
        model, tokenizers = load_esm3(device=device)
        sae_layer = 33
        tok_ref = tokenizers

    # Load SAE for activation magnitude metric
    sae = load_sae(args.model, sae_layer, "S")
    if sae is not None:
        sae = sae.to(device)
        log.info(f"Loaded SAE at layer {sae_layer}")
    else:
        log.warning(f"No SAE found at layer {sae_layer}")

    # Setup hook manager for hidden state extraction
    if args.model == "esm2":
        hook_mgr = ESM2HookManager(model, layers=[sae_layer], extract_attention=False)
    else:
        hook_mgr = ESM3HookManager(model, layers=[sae_layer], extract_attention=False)

    # ================================================================
    # Main loop: per-protein importance & conservation computation
    # ================================================================
    per_protein = {}
    all_corr_entropy_vs_sae = []
    all_corr_entropy_vs_mutkl = []

    # Breakdown collectors
    breakdown = {
        "buried": {"entropy_sae": [], "entropy_mutkl": []},
        "exposed": {"entropy_sae": [], "entropy_mutkl": []},
        "functional": {"entropy_sae": [], "entropy_mutkl": []},
        "nonfunctional": {"entropy_sae": [], "entropy_mutkl": []},
    }

    t0 = time.time()
    n_done = 0

    for acc, seq in seqs.items():
        L = len(seq)
        if L < 20:
            continue

        try:
            # 1. MLM entropy (conservation proxy)
            if args.model == "esm2":
                mlm_entropy = compute_mlm_entropy_esm2(model, tokenizer, seq, device)
            else:
                mlm_entropy = compute_mlm_entropy_esm3(model, tokenizers, seq, device)

            # 2. SAE activation magnitude
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

                h = hook_mgr.cache.residual_stream[sae_layer][0][1:-1].float()  # (L, d)
                sae_mag = compute_sae_activation_magnitude(sae, h, device)

            # 3. Mutation KL (subset of positions)
            rng = np.random.RandomState(hash(acc) % 2**31)
            n_pos = min(args.mutation_positions_per_protein, L)
            mut_positions = sorted(rng.choice(L, n_pos, replace=False).tolist())
            mut_kl = compute_mutation_kl_subset(
                model, tok_ref, seq, mut_positions, args.model, device)

            # Compute correlations
            # Entropy vs SAE (full sequence)
            rho_ent_sae, p_ent_sae = scipy_stats.spearmanr(mlm_entropy, sae_mag)

            # Entropy vs mutation KL (subset positions only)
            ent_sub = mlm_entropy[mut_positions]
            kl_sub = np.array([mut_kl[p] for p in mut_positions])
            if np.std(ent_sub) > 0 and np.std(kl_sub) > 0:
                rho_ent_kl, p_ent_kl = scipy_stats.spearmanr(ent_sub, kl_sub)
            else:
                rho_ent_kl, p_ent_kl = 0.0, 1.0

            per_protein[acc] = {
                "n_residues": L,
                "entropy_vs_sae": {"spearman_rho": float(rho_ent_sae),
                                   "p_value": float(p_ent_sae)},
                "entropy_vs_mutation_kl": {"spearman_rho": float(rho_ent_kl),
                                           "p_value": float(p_ent_kl)},
                "mean_mlm_entropy": float(np.mean(mlm_entropy)),
                "mean_sae_magnitude": float(np.mean(sae_mag)),
                "mean_mutation_kl": float(np.mean(list(mut_kl.values()))),
            }

            if not np.isnan(rho_ent_sae):
                all_corr_entropy_vs_sae.append(rho_ent_sae)
            if not np.isnan(rho_ent_kl):
                all_corr_entropy_vs_mutkl.append(rho_ent_kl)

            # Breakdown by residue category
            dssp_data = dssp.get(acc, [])
            func_set = functional_sites.get(acc, set())

            for pos in range(L):
                ent_val = mlm_entropy[pos]
                sae_val = sae_mag[pos]

                if pos in func_set:
                    breakdown["functional"]["entropy_sae"].append((ent_val, sae_val))
                else:
                    breakdown["nonfunctional"]["entropy_sae"].append((ent_val, sae_val))

                if pos < len(dssp_data):
                    asa = dssp_data[pos].get("asa", 50)
                    if asa < 25:
                        breakdown["buried"]["entropy_sae"].append((ent_val, sae_val))
                    else:
                        breakdown["exposed"]["entropy_sae"].append((ent_val, sae_val))

            n_done += 1
            if n_done % 10 == 0:
                mean_rho = np.mean(all_corr_entropy_vs_sae) if all_corr_entropy_vs_sae else 0
                log.info(f"  [{n_done}/{len(seqs)} proteins, {time.time()-t0:.0f}s] "
                         f"mean ρ(entropy,SAE)={mean_rho:.3f}")

        except Exception as e:
            log.warning(f"  {acc}: {e}")
            torch.cuda.empty_cache()

    log.info(f"\nCompleted {n_done} proteins in {time.time()-t0:.0f}s")

    # ================================================================
    # Aggregate statistics
    # ================================================================
    def agg(values, name):
        if not values:
            return {"mean": 0, "std": 0, "n": 0}
        arr = np.array(values)
        ci_lo, ci_hi = np.percentile(arr, [2.5, 97.5])
        log.info(f"  {name}: mean={np.mean(arr):.4f} ± {np.std(arr):.4f} "
                 f"[{ci_lo:.4f}, {ci_hi:.4f}] (n={len(arr)})")
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "ci_95_lo": float(ci_lo),
            "ci_95_hi": float(ci_hi),
            "n": len(arr),
            "n_significant": int(np.sum(np.abs(arr) > 0.1)),
        }

    log.info("=== Aggregate Results ===")
    aggregate = {
        "entropy_vs_sae": agg(all_corr_entropy_vs_sae, "MLM entropy vs SAE magnitude"),
        "entropy_vs_mutation_kl": agg(all_corr_entropy_vs_mutkl, "MLM entropy vs mutation KL"),
    }

    # Breakdown correlations
    breakdown_results = {}
    for cat, data in breakdown.items():
        pairs = data["entropy_sae"]
        if len(pairs) > 100:
            ent_vals = np.array([p[0] for p in pairs])
            sae_vals = np.array([p[1] for p in pairs])
            rho, p = scipy_stats.spearmanr(ent_vals, sae_vals)
            breakdown_results[cat] = {
                "entropy_vs_sae_spearman": float(rho),
                "p_value": float(p),
                "n_residues": len(pairs),
                "mean_entropy": float(np.mean(ent_vals)),
                "mean_sae_mag": float(np.mean(sae_vals)),
            }
            log.info(f"  {cat}: ρ(entropy,SAE)={rho:.4f} (n={len(pairs)})")

    output = {
        "model": args.model,
        "n_proteins": n_done,
        "sae_layer": sae_layer,
        "conservation_proxy": "mlm_entropy",
        "aggregate": aggregate,
        "breakdown": breakdown_results,
        "per_protein": per_protein,
    }

    out_path = out_dir / "evolutionary_correlation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
