#!/usr/bin/env python3
"""Head ablation analysis for ESM-2 and ESM-3.

At selected layers, zero each attention head individually and measure
prediction change (KL divergence).

Run as:
    ./env/bin/python scripts/unified/run_head_ablation.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_head_ablation.py --model esm3 --device cuda:1

ESM-2: 5 layers x 20 heads = 100 ablations
ESM-3: 12 layers x 24 heads = 288 ablations

Output: results/unified/{model}/head_ablation.json
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

from models.metrics import mean_kl_divergence
from models.interventions import head_ablation, get_num_heads


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "head_ablation_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("head_ablation"), out_dir


def load_sequences(max_proteins=200, max_length=500):
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


def get_ablation_layers(model_type, n_layers):
    """Select layers for head ablation (representative subset)."""
    if model_type == "esm2":
        # 33 layers -> 5 representative layers
        return [0, 8, 16, 24, 32]
    else:
        # 48 layers -> 12 representative layers
        return [0, 2, 4, 6, 8, 12, 16, 24, 32, 38, 44, 47]


def run_esm2_head_ablation(model, tokenizer, sequences, device, log):
    """Run head ablation for ESM-2."""
    n_layers = len(model.esm.encoder.layer)
    n_heads = get_num_heads(model, "esm2")
    ablation_layers = get_ablation_layers("esm2", n_layers)
    eval_seqs = list(sequences.items())

    log.info(f"ESM-2: {n_layers} layers, {n_heads} heads, ablating {len(ablation_layers)} layers")

    # Pre-compute baseline logits
    log.info("Computing baseline logits...")
    baseline_logits = []
    for acc, seq in eval_seqs:
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = model(**inputs)
        baseline_logits.append(out.logits.cpu())

    # Ablate each head
    results = {}
    for layer_idx in ablation_layers:
        for head_idx in range(n_heads):
            t0 = time.time()
            kl_divs = []
            for seq_idx, (acc, seq) in enumerate(eval_seqs):
                inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
                with head_ablation(model, "esm2", layer_idx, head_idx):
                    with torch.no_grad():
                        out_abl = model(**inputs)
                kl = mean_kl_divergence(baseline_logits[seq_idx], out_abl.logits.cpu())
                kl_divs.append(kl)

            key = f"L{layer_idx}_H{head_idx}"
            results[key] = {
                "mean_kl": float(np.mean(kl_divs)),
                "std_kl": float(np.std(kl_divs)),
                "layer": layer_idx,
                "head": head_idx,
            }
            dt = time.time() - t0
            log.info(f"  {key}: KL={np.mean(kl_divs):.4f} ({dt:.1f}s)")

    return results, n_heads, ablation_layers


def run_esm3_head_ablation(model, tokenizers, sequences, device, log):
    """Run head ablation for ESM-3."""
    n_layers = len(model.transformer.blocks)
    n_heads = get_num_heads(model, "esm3")
    ablation_layers = get_ablation_layers("esm3", n_layers)
    eval_seqs = list(sequences.items())

    log.info(f"ESM-3: {n_layers} layers, {n_heads} heads, ablating {len(ablation_layers)} layers")

    # Pre-compute baseline logits
    log.info("Computing baseline logits...")
    baseline_logits = []
    for acc, seq in eval_seqs:
        seq_tokens = tokenizers.sequence.encode(seq)
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(sequence_tokens=seq_tensor)
        logits = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
        baseline_logits.append(logits.float().cpu())

    # Ablate each head
    results = {}
    for layer_idx in ablation_layers:
        for head_idx in range(n_heads):
            t0 = time.time()
            kl_divs = []
            for seq_idx, (acc, seq) in enumerate(eval_seqs):
                seq_tokens = tokenizers.sequence.encode(seq)
                seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

                with head_ablation(model, "esm3", layer_idx, head_idx):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out_abl = model(sequence_tokens=seq_tensor)

                logits_abl = out_abl.sequence_logits if hasattr(out_abl, 'sequence_logits') else out_abl.logits
                kl = mean_kl_divergence(baseline_logits[seq_idx], logits_abl.float().cpu())
                kl_divs.append(kl)

            key = f"L{layer_idx}_H{head_idx}"
            results[key] = {
                "mean_kl": float(np.mean(kl_divs)),
                "std_kl": float(np.std(kl_divs)),
                "layer": layer_idx,
                "head": head_idx,
            }
            dt = time.time() - t0
            log.info(f"  {key}: KL={np.mean(kl_divs):.4f} ({dt:.1f}s)")

    return results, n_heads, ablation_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Head Ablation: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        per_head, n_heads, ablation_layers = run_esm2_head_ablation(
            model, tokenizer, sequences, args.device, log
        )
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        per_head, n_heads, ablation_layers = run_esm3_head_ablation(
            model, tokenizers, sequences, args.device, log
        )

    output = {
        "model": args.model,
        "n_heads": n_heads,
        "ablation_layers": ablation_layers,
        "n_proteins": len(sequences),
        "per_head": per_head,
    }

    out_path = out_dir / "head_ablation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
