#!/usr/bin/env python3
"""Logit lens analysis for ESM-2 and ESM-3.

At each transformer layer, project the residual stream through the LM head
to get intermediate "predictions". Measures when the model forms its final
predictions by comparing intermediate logits against final logits.

For ESM-3, also projects through structure and function heads separately
to see when each modality prediction crystallizes.

Run as:
    ./env/bin/python scripts/unified/run_logit_lens.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_logit_lens.py --model esm3 --device cuda:1

Output: results/unified/{model}/logit_lens.json
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
    get_layers, get_lm_head, get_final_norm,
    logit_lens_project, cache_residual_stream
)


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "logit_lens_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("logit_lens"), out_dir


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


def run_logit_lens_esm2(model, tokenizer, sequences, device, log):
    """Run logit lens for ESM-2 across all layers."""
    n_layers = len(model.esm.encoder.layer)
    all_layers = list(range(n_layers))
    eval_seqs = list(sequences.items())

    per_layer = {l: {"kl_divs": [], "top1_matches": [], "top5_matches": []}
                 for l in all_layers}

    for seq_idx, (acc, seq) in enumerate(eval_seqs):
        inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with cache_residual_stream(model, "esm2", all_layers) as cache:
            with torch.no_grad():
                out = model(**inputs)
        final_logits = out.logits.cpu()
        final_preds = final_logits.argmax(dim=-1)  # (1, L)
        final_top5 = final_logits.topk(5, dim=-1).indices  # (1, L, 5)

        for layer_idx in all_layers:
            hidden = cache[layer_idx].to(device)
            intermediate_logits = logit_lens_project(hidden, model, "esm2").cpu()

            kl = mean_kl_divergence(final_logits, intermediate_logits)
            per_layer[layer_idx]["kl_divs"].append(kl)

            # Top-1 agreement with final predictions
            inter_preds = intermediate_logits.argmax(dim=-1)
            match_1 = (inter_preds == final_preds).float().mean().item()
            per_layer[layer_idx]["top1_matches"].append(match_1)

            # Top-5 agreement
            inter_top5 = intermediate_logits.topk(5, dim=-1).indices
            match_5 = 0.0
            for k in range(5):
                match_5 += (inter_top5 == final_preds.unsqueeze(-1)).any(dim=-1).float().mean().item()
            match_5 = (inter_top5 == final_preds.unsqueeze(-1)).any(dim=-1).float().mean().item()
            per_layer[layer_idx]["top5_matches"].append(match_5)

        if (seq_idx + 1) % 50 == 0:
            log.info(f"  [{seq_idx+1}/{len(eval_seqs)}]")

    results = {}
    for l in all_layers:
        results[l] = {
            "mean_kl": float(np.mean(per_layer[l]["kl_divs"])),
            "std_kl": float(np.std(per_layer[l]["kl_divs"])),
            "mean_top1_agreement": float(np.mean(per_layer[l]["top1_matches"])),
            "mean_top5_agreement": float(np.mean(per_layer[l]["top5_matches"])),
        }

    return results, n_layers


def run_logit_lens_esm3(model, tokenizers, sequences, device, log):
    """Run logit lens for ESM-3 across all layers."""
    n_layers = len(model.transformer.blocks)
    all_layers = list(range(n_layers))
    eval_seqs = list(sequences.items())

    per_layer = {l: {"kl_divs": [], "top1_matches": [], "top5_matches": []}
                 for l in all_layers}

    for seq_idx, (acc, seq) in enumerate(eval_seqs):
        seq_tokens = tokenizers.sequence.encode(seq)
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with cache_residual_stream(model, "esm3", all_layers) as cache:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(sequence_tokens=seq_tensor)

        final_logits = out.sequence_logits.float().cpu() if hasattr(out, 'sequence_logits') else out.logits.float().cpu()
        final_preds = final_logits.argmax(dim=-1)

        for layer_idx in all_layers:
            hidden = cache[layer_idx].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                intermediate_logits = logit_lens_project(hidden, model, "esm3").float().cpu()

            kl = mean_kl_divergence(final_logits, intermediate_logits)
            per_layer[layer_idx]["kl_divs"].append(kl)

            inter_preds = intermediate_logits.argmax(dim=-1)
            match_1 = (inter_preds == final_preds).float().mean().item()
            per_layer[layer_idx]["top1_matches"].append(match_1)

            match_5 = 0.0
            inter_top5 = intermediate_logits.topk(5, dim=-1).indices
            match_5 = (inter_top5 == final_preds.unsqueeze(-1)).any(dim=-1).float().mean().item()
            per_layer[layer_idx]["top5_matches"].append(match_5)

        if (seq_idx + 1) % 50 == 0:
            log.info(f"  [{seq_idx+1}/{len(eval_seqs)}]")

    results = {}
    for l in all_layers:
        results[l] = {
            "mean_kl": float(np.mean(per_layer[l]["kl_divs"])),
            "std_kl": float(np.std(per_layer[l]["kl_divs"])),
            "mean_top1_agreement": float(np.mean(per_layer[l]["top1_matches"])),
            "mean_top5_agreement": float(np.mean(per_layer[l]["top5_matches"])),
        }

    return results, n_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Logit Lens: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    t0 = time.time()
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        results, n_layers = run_logit_lens_esm2(model, tokenizer, sequences, args.device, log)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        results, n_layers = run_logit_lens_esm3(model, tokenizers, sequences, args.device, log)

    log.info(f"Logit lens done in {time.time()-t0:.1f}s")

    # Log key results
    for l in sorted(results.keys()):
        r = results[l]
        if l % max(1, n_layers // 10) == 0 or l == n_layers - 1:
            log.info(f"  Layer {l}: KL={r['mean_kl']:.4f}, "
                     f"top1={r['mean_top1_agreement']:.3f}, "
                     f"top5={r['mean_top5_agreement']:.3f}")

    output = {
        "model": args.model,
        "n_layers": n_layers,
        "n_proteins": len(sequences),
        "per_layer": {str(k): v for k, v in results.items()},
    }

    out_path = out_dir / "logit_lens.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
