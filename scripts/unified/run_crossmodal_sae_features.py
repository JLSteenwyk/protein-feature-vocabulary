#!/usr/bin/env python3
"""Cross-modal SAE feature analysis for ESM-3 (with ESM-2 negative control).

Identifies SAE features whose activations change when structure tokens
are provided vs sequence-only. These "structure-modulated" features
reveal which learned concepts depend on structural information.

For ESM-3:
  - Run each SAE (S-trained and S+St-trained) on activations from BOTH conditions
  - A feature is "structure-modulated" if its activation significantly differs
  - Categorize: enhanced, suppressed, or invariant

For ESM-2 (negative control):
  - Run SAE on same sequence twice, confirm no spurious modulation

Run as:
    ./env/bin/python scripts/unified/run_crossmodal_sae_features.py --model esm3 --device cuda:1
    ./env/bin/python scripts/unified/run_crossmodal_sae_features.py --model esm2 --device cuda:0

Output: results/unified/{model}/crossmodal_sae_features.json
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
            logging.FileHandler(out_dir / "crossmodal_sae_features_log.txt", mode='w'),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("crossmodal"), out_dir


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
    parser.add_argument("--max-proteins", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=400)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Cross-Modal SAE Feature Analysis: {args.model} ===")
    device = args.device

    # Load sequences
    seq_file = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    with open(seq_file) as f:
        all_seqs = json.load(f)
    seqs = {k: v for k, v in all_seqs.items() if len(v) <= args.max_length}

    if args.model == "esm3":
        # Need proteins with structures
        struct_dir = ROOT / "data" / "scaled" / "structures"
        seqs = {k: v for k, v in seqs.items()
                if (struct_dir / f"{k}.pdb").exists()}

    sorted_accs = sorted(seqs.keys())[:args.max_proteins]
    seqs = {k: seqs[k] for k in sorted_accs}
    log.info(f"Selected {len(seqs)} proteins")

    # Define layers and SAE configs
    if args.model == "esm2":
        layers = [16, 24]
        sae_configs = [(l, "S") for l in layers]  # ESM-2 only has S
    else:
        layers = [16, 33, 42]
        sae_configs = []
        for l in layers:
            sae_configs.append((l, "S"))
            sae_configs.append((l, "S_St"))

    # Load all SAEs
    saes = {}
    for layer, cond in sae_configs:
        sae = load_sae(args.model, layer, cond)
        if sae is not None:
            sae = sae.to(device)
            saes[(layer, cond)] = sae
            log.info(f"  Loaded SAE: layer={layer}, condition={cond}, "
                     f"dict_size={sae.config.dict_size}")
        else:
            log.warning(f"  Missing SAE: layer={layer}, condition={cond}")

    if not saes:
        log.error("No SAEs loaded, exiting")
        return

    # Load model
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=device)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=device)

    # ================================================================
    # Extract activations under two conditions
    # ================================================================
    from models.esm3_hooks import ESM3HookManager
    from models.esm2_hooks import ESM2HookManager

    unique_layers = sorted(set(l for l, _ in saes.keys()))

    # activations[condition][layer] = list of (L, d_model) tensors
    acts = {"cond_A": {l: [] for l in unique_layers},
            "cond_B": {l: [] for l in unique_layers}}

    t0 = time.time()
    n_done = 0

    if args.model == "esm3":
        from esm.sdk.api import ESMProtein
        struct_dir = ROOT / "data" / "scaled" / "structures"
        hook_mgr = ESM3HookManager(model, layers=unique_layers, extract_attention=False)

        for acc, seq in seqs.items():
            pdb_path = str(struct_dir / f"{acc}.pdb")
            try:
                protein_struct = ESMProtein.from_pdb(pdb_path)
                pdb_seq = protein_struct.sequence
                if pdb_seq is None or len(pdb_seq) < 10:
                    continue

                # Condition A: sequence only
                protein_s = ESMProtein(sequence=pdb_seq)
                tok_s = model.encode(protein_s)

                # Condition B: sequence + structure
                tok_sst = model.encode(protein_struct)

                for cond_name, tok in [("cond_A", tok_s), ("cond_B", tok_sst)]:
                    kwargs = {}
                    kwargs["sequence_tokens"] = tok.sequence.unsqueeze(0).to(device)
                    if tok.structure is not None:
                        kwargs["structure_tokens"] = tok.structure.unsqueeze(0).to(device)
                    if tok.function is not None:
                        kwargs["function_tokens"] = tok.function.unsqueeze(0).to(device)

                    hook_mgr.cache.clear()
                    hook_mgr.register()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        model(**kwargs)
                    hook_mgr.remove()

                    for l in unique_layers:
                        h = hook_mgr.cache.residual_stream[l][0]  # (L_tok, d)
                        # Strip BOS/EOS tokens
                        h = h[1:-1].float().cpu()
                        acts[cond_name][l].append(h)

                n_done += 1
                if n_done % 25 == 0:
                    log.info(f"  [{n_done}/{len(seqs)} proteins, {time.time()-t0:.0f}s]")

            except Exception as e:
                if n_done < 5:
                    log.warning(f"  {acc}: {e}")

    else:
        # ESM-2: both conditions are identical (negative control)
        hook_mgr = ESM2HookManager(model, layers=unique_layers, extract_attention=False)

        for acc, seq in seqs.items():
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)

            for cond_name in ["cond_A", "cond_B"]:
                hook_mgr.cache.clear()
                hook_mgr.register()
                with torch.no_grad():
                    model(**inputs)
                hook_mgr.remove()

                for l in unique_layers:
                    h = hook_mgr.cache.residual_stream[l][0]  # (L_tok, d)
                    h = h[1:-1].float().cpu()
                    acts[cond_name][l].append(h)

            n_done += 1
            if n_done % 50 == 0:
                log.info(f"  [{n_done}/{len(seqs)} proteins, {time.time()-t0:.0f}s]")

    log.info(f"Extracted activations: {n_done} proteins in {time.time()-t0:.0f}s")

    # ================================================================
    # Encode through SAEs and compute modulation scores
    # ================================================================
    output = {
        "model": args.model,
        "n_proteins": n_done,
        "condition_A": "S" if args.model == "esm3" else "seq_run1",
        "condition_B": "S+St" if args.model == "esm3" else "seq_run2",
        "per_layer": {},
    }

    for (layer, sae_cond), sae in saes.items():
        log.info(f"\n=== Layer {layer}, SAE condition: {sae_cond} ===")

        # Concatenate all positions for each condition
        acts_A = torch.cat(acts["cond_A"][layer], dim=0)  # (N_total, d_model)
        acts_B = torch.cat(acts["cond_B"][layer], dim=0)

        n_positions = acts_A.shape[0]
        dict_size = sae.config.dict_size
        log.info(f"  {n_positions} positions, dict_size={dict_size}")

        # Encode in batches
        batch_size = 4096
        z_A_list, z_B_list = [], []
        for i in range(0, n_positions, batch_size):
            with torch.no_grad():
                z_A_list.append(sae.encode(acts_A[i:i+batch_size].to(device)).cpu())
                z_B_list.append(sae.encode(acts_B[i:i+batch_size].to(device)).cpu())

        z_A = torch.cat(z_A_list, dim=0).numpy()  # (N, dict_size)
        z_B = torch.cat(z_B_list, dim=0).numpy()

        # Per-feature modulation analysis
        mean_A = z_A.mean(axis=0)
        mean_B = z_B.mean(axis=0)
        std_pool = np.sqrt((z_A.var(axis=0) + z_B.var(axis=0)) / 2 + 1e-10)

        cohens_d = (mean_B - mean_A) / std_pool
        activation_ratio = (mean_B + 1e-8) / (mean_A + 1e-8)

        # Fraction of positions where feature fires (>0)
        fire_A = (z_A > 0).mean(axis=0)
        fire_B = (z_B > 0).mean(axis=0)
        fire_diff = fire_B - fire_A

        # Classify features
        d_threshold = 0.5
        # Bonferroni-corrected significance
        p_threshold = 0.001 / dict_size

        n_enhanced = 0
        n_suppressed = 0
        n_invariant = 0
        top_features = []

        for f_idx in range(dict_size):
            d = cohens_d[f_idx]
            # Only do expensive Wilcoxon test for features with large effect
            if abs(d) > d_threshold:
                # Subsample for speed if needed
                if n_positions > 5000:
                    rng = np.random.RandomState(f_idx)
                    idx = rng.choice(n_positions, 5000, replace=False)
                    a_vals = z_A[idx, f_idx]
                    b_vals = z_B[idx, f_idx]
                else:
                    a_vals = z_A[:, f_idx]
                    b_vals = z_B[:, f_idx]

                # Only test if there's variance
                diff = b_vals - a_vals
                if np.std(diff) < 1e-10:
                    n_invariant += 1
                    continue

                try:
                    stat, pval = scipy_stats.wilcoxon(a_vals, b_vals)
                except ValueError:
                    pval = 1.0

                if pval < p_threshold:
                    category = "enhanced" if d > 0 else "suppressed"
                    if d > 0:
                        n_enhanced += 1
                    else:
                        n_suppressed += 1
                    top_features.append({
                        "feature_idx": int(f_idx),
                        "cohens_d": float(d),
                        "p_value": float(pval),
                        "mean_act_A": float(mean_A[f_idx]),
                        "mean_act_B": float(mean_B[f_idx]),
                        "fire_rate_A": float(fire_A[f_idx]),
                        "fire_rate_B": float(fire_B[f_idx]),
                        "category": category,
                    })
                else:
                    n_invariant += 1
            else:
                n_invariant += 1

        # Sort by absolute effect size
        top_features.sort(key=lambda x: abs(x["cohens_d"]), reverse=True)

        n_modulated = n_enhanced + n_suppressed
        layer_key = f"L{layer}_{sae_cond}"
        output["per_layer"][layer_key] = {
            "layer": layer,
            "sae_condition": sae_cond,
            "dict_size": dict_size,
            "n_positions": n_positions,
            "n_structure_enhanced": n_enhanced,
            "n_structure_suppressed": n_suppressed,
            "n_structure_invariant": n_invariant,
            "n_modulated": n_modulated,
            "fraction_modulated": n_modulated / dict_size if dict_size > 0 else 0,
            "top_modulated_features": top_features[:50],
            "mean_abs_cohens_d": float(np.mean(np.abs(cohens_d))),
            "median_abs_cohens_d": float(np.median(np.abs(cohens_d))),
        }

        log.info(f"  Enhanced: {n_enhanced}, Suppressed: {n_suppressed}, "
                 f"Invariant: {n_invariant}")
        log.info(f"  Fraction modulated: {n_modulated/dict_size:.3f}")
        log.info(f"  Mean |Cohen's d|: {np.mean(np.abs(cohens_d)):.4f}")

    # Summary across layers
    if args.model == "esm3":
        layer_prog = {}
        for key, data in output["per_layer"].items():
            l = data["layer"]
            if l not in layer_prog:
                layer_prog[l] = {}
            layer_prog[l][data["sae_condition"]] = {
                "fraction_modulated": data["fraction_modulated"],
                "n_enhanced": data["n_structure_enhanced"],
                "n_suppressed": data["n_structure_suppressed"],
                "mean_abs_cohens_d": data["mean_abs_cohens_d"],
            }
        output["layer_progression"] = layer_prog

    out_path = out_dir / "crossmodal_sae_features.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
