#!/usr/bin/env python3
"""Layer ablation under S vs S+St conditions for ESM-3.

Mean-ablates each layer and measures KL divergence from the unablated output,
under both sequence-only (S) and sequence+structure (S+St) conditions.

This reveals which layers become more or less critical when structure tokens
are provided — complementing CKA modality dropout with a causal perspective.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/38_layer_ablation_sst.py
"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("layer_abl_sst")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
N_PROTEINS = 100
SEED = 42


def mean_kl(logits_orig, logits_ablated):
    min_L = min(logits_orig.size(1), logits_ablated.size(1))
    log_p = F.log_softmax(logits_orig[:, :min_L].float(), dim=-1)
    log_q = F.log_softmax(logits_ablated[:, :min_L].float(), dim=-1)
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum(dim=-1).mean())


def main():
    from models.esm3_hooks import load_esm3
    from models.interventions import layer_ablation, cache_residual_stream
    from esm.sdk.api import ESMProtein

    with open(STRUCT_DIR / "manifest.json") as f:
        available = json.load(f)["available"]
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and 50 <= len(sequences[acc]) <= 400]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins with structures")

    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")
    n_layers = len(model.transformer.blocks)
    log.info(f"Testing all {n_layers} layers under S and S+St conditions")

    # Pre-compute reference means for mean ablation (from S-only, first 50 proteins)
    log.info("Computing reference means for mean ablation...")
    ref_sums = {l: None for l in range(n_layers)}
    ref_counts = {l: 0 for l in range(n_layers)}
    n_ref = min(50, len(selected))

    for idx, acc in enumerate(selected[:n_ref]):
        try:
            seq = sequences[acc]
            prot_s = ESMProtein(sequence=seq)
            tok_s = model.encode(prot_s)
            seq_tokens = tok_s.sequence.to("cuda").unsqueeze(0)

            with cache_residual_stream(model, "esm3", list(range(n_layers))) as cache:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(sequence_tokens=seq_tokens)
            for l, tensor in cache.items():
                mean_val = tensor.float().mean(dim=(0, 1))
                if ref_sums[l] is None:
                    ref_sums[l] = mean_val
                else:
                    ref_sums[l] += mean_val
                ref_counts[l] += 1
        except Exception:
            continue
        if (idx + 1) % 25 == 0:
            log.info(f"  Ref means: {idx+1}/{n_ref}")

    ref_means = {}
    for l in range(n_layers):
        if ref_sums[l] is not None:
            ref_means[l] = (ref_sums[l] / ref_counts[l]).unsqueeze(0).unsqueeze(0)

    log.info(f"Computed reference means for {len(ref_means)} layers")

    # Pre-compute baseline logits for both conditions
    log.info("Computing baselines...")
    baselines_s = []
    baselines_sst = []
    valid_proteins = []

    for acc in selected:
        try:
            prot_sst = ESMProtein.from_pdb(available[acc])
            if prot_sst.sequence is None:
                continue
            seq = prot_sst.sequence
            if len(seq) < 50 or len(seq) > 400:
                continue

            prot_s = ESMProtein(sequence=seq)
            tok_s = model.encode(prot_s)
            tok_sst = model.encode(prot_sst)
            seq_tokens = tok_s.sequence.to("cuda").unsqueeze(0)
            struct_tokens = tok_sst.structure.to("cuda").unsqueeze(0)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_s = model(sequence_tokens=seq_tokens)
                out_sst = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)

            baselines_s.append((seq_tokens.cpu(), out_s.sequence_logits.cpu()))
            baselines_sst.append((seq_tokens.cpu(), struct_tokens.cpu(), out_sst.sequence_logits.cpu()))
            valid_proteins.append(acc)
        except Exception:
            continue

    log.info(f"  {len(valid_proteins)} proteins with valid baselines")

    # Ablate each layer under both conditions
    results_s = {}
    results_sst = {}
    t0 = time.time()

    for layer_idx in range(n_layers):
        kl_s_list = []
        kl_sst_list = []

        for pi in range(len(valid_proteins)):
            seq_tokens = baselines_s[pi][0].to("cuda")
            baseline_logits_s = baselines_s[pi][1]

            seq_tokens_sst = baselines_sst[pi][0].to("cuda")
            struct_tokens = baselines_sst[pi][1].to("cuda")
            baseline_logits_sst = baselines_sst[pi][2]

            # S-only ablation
            with layer_ablation(model, "esm3", layer_idx, method="mean",
                                ref_mean=ref_means[layer_idx]):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_abl_s = model(sequence_tokens=seq_tokens)
            kl_s = mean_kl(baseline_logits_s, out_abl_s.sequence_logits.cpu())
            kl_s_list.append(kl_s)

            # S+St ablation
            with layer_ablation(model, "esm3", layer_idx, method="mean",
                                ref_mean=ref_means[layer_idx]):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_abl_sst = model(sequence_tokens=seq_tokens_sst, structure_tokens=struct_tokens)
            kl_sst = mean_kl(baseline_logits_sst, out_abl_sst.sequence_logits.cpu())
            kl_sst_list.append(kl_sst)

        results_s[str(layer_idx)] = {
            "mean_kl": float(np.mean(kl_s_list)),
            "std_kl": float(np.std(kl_s_list)),
            "median_kl": float(np.median(kl_s_list)),
        }
        results_sst[str(layer_idx)] = {
            "mean_kl": float(np.mean(kl_sst_list)),
            "std_kl": float(np.std(kl_sst_list)),
            "median_kl": float(np.median(kl_sst_list)),
        }

        elapsed = time.time() - t0
        log.info(f"  Layer {layer_idx}/{n_layers}: S={np.mean(kl_s_list):.4f}, "
                 f"S+St={np.mean(kl_sst_list):.4f}, "
                 f"diff={np.mean(kl_sst_list)-np.mean(kl_s_list):.4f} "
                 f"({elapsed/60:.1f} min)")

    output = {
        "n_proteins": len(valid_proteins),
        "n_layers": n_layers,
        "n_ref": n_ref,
        "seed": SEED,
        "condition_s": {"per_layer": results_s},
        "condition_sst": {"per_layer": results_sst},
    }

    out_path = OUTPUT_DIR / "layer_ablation_sst.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

    # Summary
    layers = sorted(results_s.keys(), key=int)
    diffs = [results_sst[l]["mean_kl"] - results_s[l]["mean_kl"] for l in layers]
    peak_layer = layers[np.argmax(np.abs(diffs))]
    log.info(f"\nLargest |diff| at layer {peak_layer}: "
             f"S={results_s[peak_layer]['mean_kl']:.4f}, "
             f"S+St={results_sst[peak_layer]['mean_kl']:.4f}, "
             f"diff={diffs[int(peak_layer)]:.4f}")


if __name__ == "__main__":
    main()
