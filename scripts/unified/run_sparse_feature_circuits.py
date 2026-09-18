#!/usr/bin/env python3
"""Sparse feature circuits: SAE feature → feature causal graph.

Traces causal connections between SAE features across layers. For each
downstream feature at a later layer, identifies which upstream features
at an earlier layer causally influence it.

Method:
1. Load trained SAEs at two layers (upstream and downstream)
2. For each downstream feature (top-200 by activation frequency):
   a. Identify proteins where it fires
   b. For each upstream feature: ablate it (zero in SAE) and measure
      change in downstream feature activation
   c. A significant change means a causal connection exists
3. Build a directed graph of feature-to-feature causal connections

For ESM-2: layers 16 → 24
For ESM-3: layers 16 → 33, 33 → 42

Run as:
    ./env/bin/python scripts/unified/run_sparse_feature_circuits.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_sparse_feature_circuits.py --model esm3 --device cuda:1

Output: results/unified/{model}/sparse_feature_circuits.json
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
from scipy import stats

from models.interventions import get_layers, sae_feature_ablation, cache_residual_stream


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "sparse_feature_circuits_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("circuits"), out_dir


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


def load_sae(model_name, layer, condition="S"):
    """Load a trained SAE checkpoint.

    Checkpoint directory structure:
      models/sae/esm2/layer_{layer}_topk/best.pt
      models/sae/esm2_scaled/layer_{layer}_topk/best.pt
      models/sae/esm3/{condition}_layer_{layer}_topk/best.pt
      models/sae/esm3_scaled/{condition}_layer_{layer}_topk/best.pt
    """
    from sae.model import TopKSAE
    sae_dir = ROOT / "models" / "sae"

    # Try multiple directory patterns in order of preference
    search_paths = []
    if model_name == "esm2":
        search_paths = [
            sae_dir / "esm2_scaled" / f"layer_{layer}_topk" / "best.pt",
            sae_dir / "esm2" / f"layer_{layer}_topk" / "best.pt",
            sae_dir / "esm2_scaled" / f"layer_{layer}_topk" / "final.pt",
            sae_dir / "esm2" / f"layer_{layer}_topk" / "final.pt",
        ]
    else:
        cond = condition.replace("+", "_")  # S_St for S+St
        search_paths = [
            sae_dir / "esm3_scaled" / f"{cond}_layer_{layer}_topk" / "best.pt",
            sae_dir / "esm3" / f"{cond}_layer_{layer}_topk" / "best.pt",
            sae_dir / "esm3_scaled" / f"{cond}_layer_{layer}_topk" / "final.pt",
            sae_dir / "esm3" / f"{cond}_layer_{layer}_topk" / "final.pt",
        ]

    ckpt_path = None
    for path in search_paths:
        if path.exists():
            ckpt_path = path
            break

    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["sae_config"]
    sae = TopKSAE(config)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def get_top_features(sae, activations, top_k=100):
    """Find top-k features by activation frequency."""
    with torch.no_grad():
        z = sae.encode(activations.float())
        active = (z > 0).float().mean(dim=0)  # (n_features,)
    topk = active.topk(top_k)
    return topk.indices.tolist(), topk.values.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=200,
                        help="Proteins to use (fewer for compute efficiency)")
    parser.add_argument("--top-downstream", type=int, default=50,
                        help="Number of downstream features to trace")
    parser.add_argument("--top-upstream", type=int, default=100,
                        help="Number of upstream features to test")
    parser.add_argument("--n-proteins-per-feature", type=int, default=20,
                        help="Proteins to test per downstream feature")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Sparse Feature Circuits: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    # Define layer pairs
    if args.model == "esm2":
        layer_pairs = [(16, 24)]
    else:
        layer_pairs = [(16, 33), (33, 42)]

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)

    all_results = {}
    t0 = time.time()

    for upstream_layer, downstream_layer in layer_pairs:
        log.info(f"\n=== Layer pair: {upstream_layer} → {downstream_layer} ===")

        # Load SAEs
        sae_up = load_sae(args.model, upstream_layer)
        sae_down = load_sae(args.model, downstream_layer)

        if sae_up is None or sae_down is None:
            log.warning(f"  Missing SAE for layer pair {upstream_layer}→{downstream_layer}, skipping")
            continue

        sae_up = sae_up.to(args.device)
        sae_down = sae_down.to(args.device)

        # First pass: collect activations at both layers to find top features
        log.info("  Collecting activations to identify top features...")
        up_acts_all = []
        down_acts_all = []

        for seq_idx, (acc, seq) in enumerate(sequences.items()):
            if seq_idx >= 100:
                break
            if len(seq) < 10:
                continue

            with cache_residual_stream(model, args.model, [upstream_layer, downstream_layer]) as cache:
                if args.model == "esm2":
                    inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                                       max_length=1024).to(args.device)
                    with torch.no_grad():
                        model(**inputs)
                else:
                    seq_tokens = tokenizers.sequence.encode(seq)
                    seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                              device=args.device).unsqueeze(0)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        model(sequence_tokens=seq_tensor)

            if upstream_layer in cache:
                up_acts_all.append(cache[upstream_layer][0, 1:-1].float())
            if downstream_layer in cache:
                down_acts_all.append(cache[downstream_layer][0, 1:-1].float())

        up_cat = torch.cat(up_acts_all, dim=0).to(args.device)
        down_cat = torch.cat(down_acts_all, dim=0).to(args.device)

        up_features, up_freqs = get_top_features(sae_up, up_cat, top_k=args.top_upstream)
        down_features, down_freqs = get_top_features(sae_down, down_cat, top_k=args.top_downstream)
        log.info(f"  Top upstream features: {up_features[:10]}... (freq: {up_freqs[0]:.3f}-{up_freqs[-1]:.3f})")
        log.info(f"  Top downstream features: {down_features[:10]}... (freq: {down_freqs[0]:.3f}-{down_freqs[-1]:.3f})")

        del up_cat, down_cat, up_acts_all, down_acts_all
        torch.cuda.empty_cache()

        # Second pass: for each downstream feature, test upstream feature influence
        log.info("  Testing causal connections...")
        pair_key = f"L{upstream_layer}_to_L{downstream_layer}"
        connections = []

        seq_list = list(sequences.items())

        for d_idx, down_feat in enumerate(down_features):
            # Find proteins where downstream feature fires
            firing_proteins = []
            for seq_idx, (acc, seq) in enumerate(seq_list[:args.max_proteins]):
                if len(seq) < 10:
                    continue
                if len(firing_proteins) >= args.n_proteins_per_feature:
                    break

                with cache_residual_stream(model, args.model, [downstream_layer]) as cache:
                    if args.model == "esm2":
                        inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                                           max_length=1024).to(args.device)
                        with torch.no_grad():
                            model(**inputs)
                    else:
                        seq_tokens = tokenizers.sequence.encode(seq)
                        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                                  device=args.device).unsqueeze(0)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            model(sequence_tokens=seq_tensor)

                if downstream_layer in cache:
                    h = cache[downstream_layer][0, 1:-1].float().to(args.device)
                    with torch.no_grad():
                        z = sae_down.encode(h)
                    if z[:, down_feat].max() > 0:
                        firing_proteins.append((acc, seq, z[:, down_feat].mean().item()))

            if len(firing_proteins) < 3:
                continue

            # For each upstream feature, ablate it and measure downstream change
            # Collect ALL upstream effects first, then apply statistical filter
            all_upstream_deltas = {}  # up_feat -> list of deltas
            for up_feat in up_features:
                deltas = []
                for acc, seq, _ in firing_proteins[:args.n_proteins_per_feature]:
                    # Baseline: get downstream feature activation
                    with cache_residual_stream(model, args.model, [downstream_layer]) as cache:
                        if args.model == "esm2":
                            inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                                               max_length=1024).to(args.device)
                            with torch.no_grad():
                                model(**inputs)
                        else:
                            seq_tokens = tokenizers.sequence.encode(seq)
                            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                                      device=args.device).unsqueeze(0)
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                model(sequence_tokens=seq_tensor)

                    baseline_h = cache[downstream_layer][0, 1:-1].float().to(args.device)
                    with torch.no_grad():
                        baseline_z = sae_down.encode(baseline_h)
                    baseline_act = baseline_z[:, down_feat].mean().item()

                    # Ablated: zero upstream feature
                    with sae_feature_ablation(model, args.model, sae_up, upstream_layer, [up_feat]):
                        with cache_residual_stream(model, args.model, [downstream_layer]) as cache2:
                            if args.model == "esm2":
                                with torch.no_grad():
                                    model(**inputs)
                            else:
                                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                    model(sequence_tokens=seq_tensor)

                    ablated_h = cache2[downstream_layer][0, 1:-1].float().to(args.device)
                    with torch.no_grad():
                        ablated_z = sae_down.encode(ablated_h)
                    ablated_act = ablated_z[:, down_feat].mean().item()

                    deltas.append(baseline_act - ablated_act)

                all_upstream_deltas[up_feat] = deltas

            # --- Statistical significance filtering ---
            # 1. Compute effect size distribution across ALL upstream features
            all_abs_means = []
            for up_feat, deltas in all_upstream_deltas.items():
                all_abs_means.append(abs(np.mean(deltas)))
            all_abs_means = np.array(all_abs_means)
            median_effect = float(np.median(all_abs_means))
            std_effect = float(np.std(all_abs_means))

            # Number of upstream features being tested (for Bonferroni)
            n_tests = len(up_features)
            bonferroni_alpha = 0.05 / n_tests  # e.g., 0.05/100 = 0.0005

            # Effect size threshold: median + 2*std of the distribution
            effect_threshold = median_effect + 2.0 * std_effect

            upstream_effects = {}
            for up_feat, deltas in all_upstream_deltas.items():
                mean_delta = float(np.mean(deltas))
                std_delta = float(np.std(deltas))
                n = len(deltas)

                # Skip if too few samples or zero variance
                if n < 3 or std_delta < 1e-12:
                    continue

                # Test 1: One-sample t-test (is the mean significantly != 0?)
                t_stat, p_value = stats.ttest_1samp(deltas, 0.0)

                # Test 2: Effect size must exceed the distribution-based threshold
                passes_effect = abs(mean_delta) > effect_threshold

                # Test 3: Bonferroni-corrected p-value
                passes_pvalue = p_value < bonferroni_alpha

                # Must pass BOTH criteria
                if passes_effect and passes_pvalue:
                    upstream_effects[up_feat] = {
                        "mean_delta": mean_delta,
                        "std_delta": std_delta,
                        "n_proteins": n,
                        "p_value": float(p_value),
                        "t_stat": float(t_stat),
                    }

            if upstream_effects:
                # Sort by effect magnitude
                sorted_effects = sorted(upstream_effects.items(),
                                        key=lambda x: abs(x[1]["mean_delta"]),
                                        reverse=True)
                connections.append({
                    "downstream_feature": down_feat,
                    "downstream_freq": down_freqs[d_idx],
                    "n_firing_proteins": len(firing_proteins),
                    "upstream_connections": {str(k): v for k, v in sorted_effects[:20]},
                    "n_significant_upstream": len(upstream_effects),
                    "effect_threshold": effect_threshold,
                    "bonferroni_alpha": bonferroni_alpha,
                    "median_effect": median_effect,
                    "std_effect": std_effect,
                })

            if (d_idx + 1) % 10 == 0:
                log.info(f"  [{d_idx+1}/{len(down_features)} downstream features, "
                         f"{len(connections)} with connections, {time.time()-t0:.0f}s]")

        all_results[pair_key] = {
            "upstream_layer": upstream_layer,
            "downstream_layer": downstream_layer,
            "n_upstream_features": len(up_features),
            "n_downstream_features": len(down_features),
            "connections": connections,
            "n_connected_downstream": len(connections),
        }
        log.info(f"  {pair_key}: {len(connections)}/{len(down_features)} downstream features have connections")

    log.info(f"\nCompleted in {time.time()-t0:.1f}s")

    output = {
        "model": args.model,
        "n_proteins": len(sequences),
        "layer_pairs": {k: v for k, v in all_results.items()},
    }

    out_path = out_dir / "sparse_feature_circuits.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
