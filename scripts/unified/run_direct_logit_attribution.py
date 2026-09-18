#!/usr/bin/env python3
"""Direct logit attribution for ESM-2 and ESM-3.

For each transformer layer, computes how much that layer's additive contribution
to the residual stream directly affects the output logits. This decomposes the
final prediction into per-layer (and per-component: attn vs MLP) contributions
in a single forward pass — no ablation needed.

Method:
1. Cache each layer's residual stream contribution (output - input)
2. Project each contribution through the unembedding matrix
3. Measure: L2 norm of logit contribution, cosine similarity with final logits,
   and fraction of total logit variance explained

Run as:
    ./env/bin/python scripts/unified/run_direct_logit_attribution.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_direct_logit_attribution.py --model esm3 --device cuda:1

Output: results/unified/{model}/direct_logit_attribution.json
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

from models.interventions import (
    get_layers, get_final_norm, get_unembedding_matrix,
    cache_residual_for_logit_attribution, cache_component_outputs,
)


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "direct_logit_attribution_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("dla"), out_dir


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Direct Logit Attribution: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        n_layers = len(model.esm.encoder.layer)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        n_layers = len(model.transformer.blocks)

    log.info(f"{args.model}: {n_layers} layers")

    # Get unembedding matrix
    W_U = get_unembedding_matrix(model, args.model)  # (V, D)
    W_U = W_U.float().to(args.device)
    log.info(f"Unembedding matrix: {W_U.shape}")

    # Get final norm
    final_norm = get_final_norm(model, args.model)

    all_layers = list(range(n_layers))

    # Per-layer accumulators
    layer_logit_norms = {l: [] for l in all_layers}
    layer_cosine_with_final = {l: [] for l in all_layers}
    layer_fraction_of_total = {l: [] for l in all_layers}

    # Per-component (attn vs MLP) accumulators — sample fewer layers
    component_layers = sorted(set([0, n_layers // 4, n_layers // 2,
                                    3 * n_layers // 4, n_layers - 1]))
    attn_logit_norms = {l: [] for l in component_layers}
    mlp_logit_norms = {l: [] for l in component_layers}

    t0 = time.time()
    n_evaluated = 0

    for seq_idx, (acc, seq) in enumerate(sequences.items()):
        if len(seq) < 10:
            continue

        # --- Phase 1: Per-layer contribution ---
        with cache_residual_for_logit_attribution(model, args.model, all_layers) as cache:
            if args.model == "esm2":
                inputs = tokenizer(seq, return_tensors="pt", truncation=True,
                                   max_length=1024).to(args.device)
                with torch.no_grad():
                    out = model(**inputs)
                final_logits = out.logits[0, 1:-1].float().cpu()  # (L, V), strip BOS/EOS
            else:
                seq_tokens = tokenizers.sequence.encode(seq)
                seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                          device=args.device).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(sequence_tokens=seq_tensor)
                logits_raw = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
                final_logits = logits_raw[0, 1:-1].float().cpu()

        # Project each layer's contribution through unembedding
        total_logit_contrib = torch.zeros_like(final_logits)
        per_layer_logit_contribs = {}

        for l in all_layers:
            contrib = cache[l].get('contribution')
            if contrib is None:
                continue
            # Strip BOS/EOS
            c = contrib[0, 1:1+final_logits.size(0)].float().to(W_U.device)
            # Apply final norm if needed (approximation: norm the contribution alone)
            # Project: (L, D) @ (D, V) -> (L, V)
            logit_c = (c @ W_U.T).cpu()
            per_layer_logit_contribs[l] = logit_c
            total_logit_contrib += logit_c

            # Metrics
            norm = logit_c.norm(dim=-1).mean().item()
            layer_logit_norms[l].append(norm)

            # Cosine sim with final logits
            cos = torch.nn.functional.cosine_similarity(
                logit_c.reshape(-1), final_logits.reshape(-1), dim=0
            ).item()
            layer_cosine_with_final[l].append(cos)

        # Fraction of total
        total_norm = total_logit_contrib.norm(dim=-1).mean().item() + 1e-10
        for l in all_layers:
            if l in per_layer_logit_contribs:
                frac = per_layer_logit_contribs[l].norm(dim=-1).mean().item() / total_norm
                layer_fraction_of_total[l].append(frac)

        # --- Phase 2: Component-level (attn vs MLP) on subset of layers ---
        with cache_component_outputs(model, args.model, component_layers) as comp_cache:
            if args.model == "esm2":
                with torch.no_grad():
                    model(**inputs)
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(sequence_tokens=seq_tensor)

        for l in component_layers:
            attn_c = comp_cache[l].get('attn')
            mlp_c = comp_cache[l].get('mlp')
            if attn_c is not None:
                a = attn_c[0, 1:1+final_logits.size(0)].float().to(W_U.device)
                attn_logit = a @ W_U.T
                attn_logit_norms[l].append(attn_logit.norm(dim=-1).mean().item())
            if mlp_c is not None:
                m = mlp_c[0, 1:1+final_logits.size(0)].float().to(W_U.device)
                mlp_logit = m @ W_U.T
                mlp_logit_norms[l].append(mlp_logit.norm(dim=-1).mean().item())

        n_evaluated += 1
        if n_evaluated % 50 == 0:
            log.info(f"  [{n_evaluated} proteins, {time.time()-t0:.0f}s]")

    log.info(f"Evaluated {n_evaluated} proteins in {time.time()-t0:.1f}s")

    # Aggregate
    def agg(d):
        return {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
                for k, v in d.items() if v}

    output = {
        "model": args.model,
        "n_layers": n_layers,
        "n_proteins": n_evaluated,
        "per_layer_logit_norm": agg(layer_logit_norms),
        "per_layer_cosine_with_final": agg(layer_cosine_with_final),
        "per_layer_fraction_of_total": agg(layer_fraction_of_total),
        "component_layers": component_layers,
        "attn_logit_norm": agg(attn_logit_norms),
        "mlp_logit_norm": agg(mlp_logit_norms),
    }

    # Log summary
    log.info("=== Summary: Top contributing layers (by logit norm) ===")
    norms = {int(k): v["mean"] for k, v in output["per_layer_logit_norm"].items()}
    for l in sorted(norms, key=norms.get, reverse=True)[:10]:
        cos = output["per_layer_cosine_with_final"].get(str(l), {}).get("mean", 0)
        log.info(f"  Layer {l}: norm={norms[l]:.4f}, cosine={cos:.4f}")

    log.info("=== Component breakdown ===")
    for l in component_layers:
        a = output["attn_logit_norm"].get(str(l), {}).get("mean", 0)
        m = output["mlp_logit_norm"].get(str(l), {}).get("mean", 0)
        log.info(f"  Layer {l}: attn={a:.4f}, mlp={m:.4f}")

    out_path = out_dir / "direct_logit_attribution.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
