#!/usr/bin/env python3
"""Layer ablation analysis for ESM-2 and ESM-3.

For each layer, mean-ablate (replace output with pre-computed mean activation)
and measure impact on output predictions.

Run as:
    ./env/bin/python scripts/unified/run_layer_ablation.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_layer_ablation.py --model esm3 --device cuda:1

Steps:
1. Pre-compute mean activations from ~100 proteins (cached to .pt file)
2. For each layer, mean-ablate and measure:
   - KL divergence of output logits vs unablated
   - Accuracy drop (sequence token prediction)

Output: results/unified/{model}/layer_ablation.json
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

from models.metrics import mean_kl_divergence, accuracy_from_logits
from models.interventions import (
    get_layers, get_lm_head, layer_ablation, cache_residual_stream
)


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "layer_ablation_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("layer_ablation"), out_dir


def load_sequences(max_proteins=200, max_length=500):
    """Load protein sequences from scaled dataset."""
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


def compute_reference_means_esm2(model, tokenizer, sequences, device, log, n_ref=100):
    """Compute mean activations per layer for ESM-2."""
    layers_list = list(range(len(model.esm.encoder.layer)))
    ref_seqs = list(sequences.values())[:n_ref]

    sums = {l: None for l in layers_list}
    counts = {l: 0 for l in layers_list}

    for i, seq in enumerate(ref_seqs):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with cache_residual_stream(model, "esm2", layers_list) as cache:
            with torch.no_grad():
                model(**inputs)
        for l, tensor in cache.items():
            mean_val = tensor.float().mean(dim=(0, 1))  # (D,)
            if sums[l] is None:
                sums[l] = mean_val
            else:
                sums[l] += mean_val
            counts[l] += 1
        if (i + 1) % 50 == 0:
            log.info(f"  Reference means: {i+1}/{len(ref_seqs)}")

    means = {}
    for l in layers_list:
        means[l] = (sums[l] / counts[l]).unsqueeze(0).unsqueeze(0)  # (1, 1, D)

    return means


def compute_reference_means_esm3(model, tokenizers, sequences, device, log, n_ref=100):
    """Compute mean activations per layer for ESM-3."""
    n_layers = len(model.transformer.blocks)
    layers_list = list(range(n_layers))
    ref_seqs = list(sequences.values())[:n_ref]

    sums = {l: None for l in layers_list}
    counts = {l: 0 for l in layers_list}

    for i, seq in enumerate(ref_seqs):
        seq_tokens = tokenizers.sequence.encode(seq)
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with cache_residual_stream(model, "esm3", layers_list) as cache:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(sequence_tokens=seq_tensor)
        for l, tensor in cache.items():
            mean_val = tensor.float().mean(dim=(0, 1))
            if sums[l] is None:
                sums[l] = mean_val
            else:
                sums[l] += mean_val
            counts[l] += 1
        if (i + 1) % 50 == 0:
            log.info(f"  Reference means: {i+1}/{len(ref_seqs)}")

    means = {}
    for l in layers_list:
        means[l] = (sums[l] / counts[l]).unsqueeze(0).unsqueeze(0)

    return means


def run_esm2_ablation(model, tokenizer, sequences, ref_means, device, log):
    """Run layer ablation for ESM-2."""
    n_layers = len(model.esm.encoder.layer)
    eval_seqs = list(sequences.items())

    # Get baseline (unablated) predictions
    log.info("Computing baseline predictions...")
    baseline_logits_list = []
    token_ids_list = []
    for acc, seq in eval_seqs:
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model(**inputs)
        baseline_logits_list.append(out.logits.cpu())
        token_ids_list.append(inputs["input_ids"].cpu())

    # Ablate each layer
    results_per_layer = {}
    for layer_idx in range(n_layers):
        t0 = time.time()
        kl_divs = []
        for seq_idx, (acc, seq) in enumerate(eval_seqs):
            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with layer_ablation(model, "esm2", layer_idx, method="mean",
                                ref_mean=ref_means[layer_idx]):
                with torch.no_grad():
                    out_ablated = model(**inputs)

            kl = mean_kl_divergence(baseline_logits_list[seq_idx].to(device),
                                     out_ablated.logits.cpu().to(device))
            kl_divs.append(kl)

        results_per_layer[layer_idx] = {
            "mean_kl": float(np.mean(kl_divs)),
            "std_kl": float(np.std(kl_divs)),
            "median_kl": float(np.median(kl_divs)),
        }
        dt = time.time() - t0
        log.info(f"  Layer {layer_idx}/{n_layers}: KL={np.mean(kl_divs):.4f} ({dt:.1f}s)")

    return results_per_layer


def run_esm3_ablation(model, tokenizers, sequences, ref_means, device, log):
    """Run layer ablation for ESM-3."""
    n_layers = len(model.transformer.blocks)
    eval_seqs = list(sequences.items())

    # Get baseline predictions
    log.info("Computing baseline predictions...")
    baseline_logits_list = []
    for acc, seq in eval_seqs:
        seq_tokens = tokenizers.sequence.encode(seq)
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(sequence_tokens=seq_tensor)
        # ESM-3 output: out.sequence_logits or out.logits
        logits = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
        baseline_logits_list.append(logits.float().cpu())

    # Ablate each layer
    results_per_layer = {}
    for layer_idx in range(n_layers):
        t0 = time.time()
        kl_divs = []
        for seq_idx, (acc, seq) in enumerate(eval_seqs):
            seq_tokens = tokenizers.sequence.encode(seq)
            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

            with layer_ablation(model, "esm3", layer_idx, method="mean",
                                ref_mean=ref_means[layer_idx]):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_ablated = model(sequence_tokens=seq_tensor)

            logits_abl = out_ablated.sequence_logits if hasattr(out_ablated, 'sequence_logits') else out_ablated.logits
            kl = mean_kl_divergence(baseline_logits_list[seq_idx],
                                     logits_abl.float().cpu())
            kl_divs.append(kl)

        results_per_layer[layer_idx] = {
            "mean_kl": float(np.mean(kl_divs)),
            "std_kl": float(np.std(kl_divs)),
            "median_kl": float(np.median(kl_divs)),
        }
        dt = time.time() - t0
        log.info(f"  Layer {layer_idx}/{n_layers}: KL={np.mean(kl_divs):.4f} ({dt:.1f}s)")

    return results_per_layer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--n-ref", type=int, default=100,
                        help="Number of proteins for reference mean computation")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Layer Ablation: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    # Check for cached reference means
    ref_cache_path = out_dir / f"reference_means.pt"

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        n_layers = len(model.esm.encoder.layer)
        log.info(f"ESM-2: {n_layers} layers")

        if ref_cache_path.exists():
            log.info(f"Loading cached reference means from {ref_cache_path}")
            ref_means = torch.load(ref_cache_path, map_location="cpu", weights_only=True)
        else:
            log.info(f"Computing reference means from {args.n_ref} proteins...")
            ref_means = compute_reference_means_esm2(model, tokenizer, sequences, args.device, log, n_ref=args.n_ref)
            torch.save(ref_means, ref_cache_path)
            log.info(f"Cached reference means to {ref_cache_path}")

        results_per_layer = run_esm2_ablation(model, tokenizer, sequences, ref_means, args.device, log)

    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        n_layers = len(model.transformer.blocks)
        log.info(f"ESM-3: {n_layers} layers")

        if ref_cache_path.exists():
            log.info(f"Loading cached reference means from {ref_cache_path}")
            ref_means = torch.load(ref_cache_path, map_location="cpu", weights_only=True)
        else:
            log.info(f"Computing reference means from {args.n_ref} proteins...")
            ref_means = compute_reference_means_esm3(model, tokenizers, sequences, args.device, log, n_ref=args.n_ref)
            torch.save(ref_means, ref_cache_path)
            log.info(f"Cached reference means to {ref_cache_path}")

        results_per_layer = run_esm3_ablation(model, tokenizers, sequences, ref_means, args.device, log)

    # Save results
    results = {
        "model": args.model,
        "n_layers": n_layers,
        "n_proteins": len(sequences),
        "n_ref_proteins": args.n_ref,
        "per_layer": {str(k): v for k, v in results_per_layer.items()},
    }

    out_path = out_dir / "layer_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
