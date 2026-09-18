#!/usr/bin/env python3
"""Residual stream decomposition for ESM-2 and ESM-3.

At each layer, decompose the residual stream update into:
- Attention sublayer contribution
- MLP/FFN sublayer contribution

Measures: L2 norm of each component, cosine similarity with the
overall layer update, and fraction of total update magnitude.

Run as:
    ./env/bin/python scripts/unified/run_residual_decomposition.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_residual_decomposition.py --model esm3 --device cuda:1

Output: results/unified/{model}/residual_decomposition.json
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

from models.interventions import get_layers, cache_component_outputs


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "residual_decomposition_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("residual_decomp"), out_dir


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


def analyze_components(cache, layers):
    """Analyze attention vs MLP contributions from cached component outputs.

    Returns per-layer metrics averaged over residues.
    """
    results = {}
    for l in layers:
        if 'attn' not in cache[l] or 'mlp' not in cache[l]:
            continue

        attn = cache[l]['attn'].float()  # (B, L, D) or (L, D)
        mlp = cache[l]['mlp'].float()

        if attn.dim() == 3:
            attn = attn.squeeze(0)
            mlp = mlp.squeeze(0)

        # L2 norms per residue, then average
        attn_norm = attn.norm(dim=-1).mean().item()
        mlp_norm = mlp.norm(dim=-1).mean().item()

        total = attn + mlp
        total_norm = total.norm(dim=-1).mean().item()

        # Fraction of total update from each component
        attn_frac = attn_norm / (attn_norm + mlp_norm + 1e-10)
        mlp_frac = mlp_norm / (attn_norm + mlp_norm + 1e-10)

        # Cosine similarity between attn and mlp
        cos_sim = torch.nn.functional.cosine_similarity(
            attn.reshape(-1, attn.size(-1)),
            mlp.reshape(-1, mlp.size(-1)),
            dim=-1
        ).mean().item()

        results[l] = {
            "attn_l2_norm": attn_norm,
            "mlp_l2_norm": mlp_norm,
            "total_l2_norm": total_norm,
            "attn_fraction": attn_frac,
            "mlp_fraction": mlp_frac,
            "attn_mlp_cosine": cos_sim,
        }
    return results


def run_esm2(model, tokenizer, sequences, device, log):
    """Run residual decomposition for ESM-2."""
    n_layers = len(model.esm.encoder.layer)
    all_layers = list(range(n_layers))
    eval_seqs = list(sequences.items())

    # Accumulate per-layer metrics across proteins
    accum = {l: {k: [] for k in [
        "attn_l2_norm", "mlp_l2_norm", "total_l2_norm",
        "attn_fraction", "mlp_fraction", "attn_mlp_cosine"
    ]} for l in all_layers}

    for seq_idx, (acc, seq) in enumerate(eval_seqs):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with cache_component_outputs(model, "esm2", all_layers) as cache:
            with torch.no_grad():
                model(**inputs)

        per_layer = analyze_components(cache, all_layers)
        for l, metrics in per_layer.items():
            for k, v in metrics.items():
                accum[l][k].append(v)

        if (seq_idx + 1) % 100 == 0:
            log.info(f"  [{seq_idx+1}/{len(eval_seqs)}]")

    # Average across proteins
    results = {}
    for l in all_layers:
        results[l] = {k: float(np.mean(v)) for k, v in accum[l].items() if v}
    return results, n_layers


def run_esm3(model, tokenizers, sequences, device, log):
    """Run residual decomposition for ESM-3."""
    n_layers = len(model.transformer.blocks)
    all_layers = list(range(n_layers))
    eval_seqs = list(sequences.items())

    accum = {l: {k: [] for k in [
        "attn_l2_norm", "mlp_l2_norm", "total_l2_norm",
        "attn_fraction", "mlp_fraction", "attn_mlp_cosine"
    ]} for l in all_layers}

    for seq_idx, (acc, seq) in enumerate(eval_seqs):
        seq_tokens = tokenizers.sequence.encode(seq)
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with cache_component_outputs(model, "esm3", all_layers) as cache:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(sequence_tokens=seq_tensor)

        per_layer = analyze_components(cache, all_layers)
        for l, metrics in per_layer.items():
            for k, v in metrics.items():
                accum[l][k].append(v)

        if (seq_idx + 1) % 100 == 0:
            log.info(f"  [{seq_idx+1}/{len(eval_seqs)}]")

    results = {}
    for l in all_layers:
        results[l] = {k: float(np.mean(v)) for k, v in accum[l].items() if v}
    return results, n_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Residual Decomposition: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    t0 = time.time()
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        results, n_layers = run_esm2(model, tokenizer, sequences, args.device, log)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        results, n_layers = run_esm3(model, tokenizers, sequences, args.device, log)

    log.info(f"Decomposition done in {time.time()-t0:.1f}s")

    for l in sorted(results.keys()):
        r = results[l]
        if l % max(1, n_layers // 8) == 0 or l == n_layers - 1:
            log.info(f"  Layer {l}: attn={r.get('attn_fraction', 0):.3f}, "
                     f"mlp={r.get('mlp_fraction', 0):.3f}, "
                     f"cos={r.get('attn_mlp_cosine', 0):.3f}")

    output = {
        "model": args.model,
        "n_layers": n_layers,
        "n_proteins": len(sequences),
        "per_layer": {str(k): v for k, v in results.items()},
    }

    out_path = out_dir / "residual_decomposition.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
