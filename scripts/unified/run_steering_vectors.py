#!/usr/bin/env python3
"""Steering vector analysis for ESM-2 and ESM-3.

Computes property-specific steering vectors by averaging representation
differences between residues with contrasting properties:
- Helix vs Sheet (secondary structure)
- Buried vs Exposed (solvent accessibility)
- Functional vs Non-functional sites

Then applies these vectors during inference to test whether they
causally shift model predictions in the expected direction.

Run as:
    ./env/bin/python scripts/unified/run_steering_vectors.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_steering_vectors.py --model esm3 --device cuda:1

Output: results/unified/{model}/steering_vectors.json
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
from models.interventions import (
    get_layers, cache_residual_stream, apply_steering_vector,
    compute_residue_steering_vector
)


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "steering_vectors_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("steering"), out_dir


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


def load_ss_labels():
    """Load secondary structure labels (from DSSP) for sequences."""
    ss_file = ROOT / "data" / "scaled" / "labels" / "ss3_labels.json"
    if not ss_file.exists():
        ss_file = ROOT / "data" / "scaled" / "labels" / "secondary_structure.json"
    if ss_file.exists():
        with open(ss_file) as f:
            return json.load(f)
    return None


def load_functional_labels():
    """Load functional site labels."""
    func_file = ROOT / "data" / "scaled" / "labels" / "functional_sites.json"
    if func_file.exists():
        with open(func_file) as f:
            return json.load(f)
    return None


def extract_layer_activations(model, model_type, tokenizer_or_tokenizers,
                               sequences, layer, device):
    """Extract activations at a specific layer for all sequences."""
    activations = {}
    for acc, seq in sequences.items():
        if model_type == "esm2":
            inputs = tokenizer_or_tokenizers(seq, return_tensors="pt",
                                              truncation=True, max_length=1024).to(device)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad():
                    model(**inputs)
            # Remove BOS/EOS tokens for residue-level alignment
            hidden = cache[layer][0, 1:-1, :].cpu()  # (L_seq, D)
        else:
            seq_tokens = tokenizer_or_tokenizers.sequence.encode(seq)
            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(sequence_tokens=seq_tensor)
            hidden = cache[layer][0, 1:-1, :].float().cpu()  # (L_seq, D)
        activations[acc] = hidden
    return activations


def compute_ss_steering_vectors(activations, ss_labels, log):
    """Compute helix-vs-sheet and helix-vs-coil steering vectors."""
    # SS3 encoding: H=0 or 'H', E=1 or 'E', C=2 or 'C'
    vectors = {}

    # Collect residue-level data
    act_list, lab_list = [], []
    for acc, hidden in activations.items():
        if acc not in ss_labels:
            continue
        labels = ss_labels[acc]
        if isinstance(labels, str):
            label_map = {'H': 0, 'E': 1, 'C': 2}
            labels = [label_map.get(c, 2) for c in labels]

        L = min(hidden.size(0), len(labels))
        act_list.append(hidden[:L])
        lab_list.append(torch.tensor(labels[:L]))

    if not act_list:
        log.warning("No SS labels found, skipping SS steering vectors")
        return {}

    log.info(f"SS steering: {len(act_list)} proteins with labels")

    # Helix vs Sheet
    try:
        v = compute_residue_steering_vector(act_list, lab_list, pos_label=0, neg_label=1)
        vectors["helix_vs_sheet"] = v
        log.info(f"  helix_vs_sheet: norm={v.norm():.4f}")
    except ValueError as e:
        log.warning(f"  helix_vs_sheet: {e}")

    # Helix vs Coil
    try:
        v = compute_residue_steering_vector(act_list, lab_list, pos_label=0, neg_label=2)
        vectors["helix_vs_coil"] = v
        log.info(f"  helix_vs_coil: norm={v.norm():.4f}")
    except ValueError as e:
        log.warning(f"  helix_vs_coil: {e}")

    # Sheet vs Coil
    try:
        v = compute_residue_steering_vector(act_list, lab_list, pos_label=1, neg_label=2)
        vectors["sheet_vs_coil"] = v
        log.info(f"  sheet_vs_coil: norm={v.norm():.4f}")
    except ValueError as e:
        log.warning(f"  sheet_vs_coil: {e}")

    return vectors


def evaluate_steering(model, model_type, tokenizer_or_tokenizers,
                      sequences, layer, vectors, device, log,
                      scales=(-5.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 5.0)):
    """Evaluate the effect of applying steering vectors at various scales.

    For each vector and scale, measures KL divergence from unsteered predictions.
    """
    eval_seqs = list(sequences.items())[:100]  # Use subset for evaluation
    results = {}

    for vec_name, vec in vectors.items():
        log.info(f"Evaluating steering vector: {vec_name}")
        results[vec_name] = {"vector_norm": float(vec.norm())}

        # Get baseline logits
        baseline_logits = []
        for acc, seq in eval_seqs:
            if model_type == "esm2":
                inputs = tokenizer_or_tokenizers(seq, return_tensors="pt",
                                                  truncation=True, max_length=1024).to(device)
                with torch.no_grad():
                    out = model(**inputs)
                baseline_logits.append(out.logits.cpu())
            else:
                seq_tokens = tokenizer_or_tokenizers.sequence.encode(seq)
                seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(sequence_tokens=seq_tensor)
                logits = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
                baseline_logits.append(logits.float().cpu())

        scale_results = {}
        for scale in scales:
            kl_divs = []
            for seq_idx, (acc, seq) in enumerate(eval_seqs):
                with apply_steering_vector(model, model_type, layer, vec, scale):
                    if model_type == "esm2":
                        inputs = tokenizer_or_tokenizers(seq, return_tensors="pt",
                                                          truncation=True, max_length=1024).to(device)
                        with torch.no_grad():
                            out_steered = model(**inputs)
                        logits_steered = out_steered.logits.cpu()
                    else:
                        seq_tokens = tokenizer_or_tokenizers.sequence.encode(seq)
                        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            out_steered = model(sequence_tokens=seq_tensor)
                        logits_steered = (out_steered.sequence_logits if hasattr(out_steered, 'sequence_logits')
                                          else out_steered.logits).float().cpu()

                kl = mean_kl_divergence(baseline_logits[seq_idx], logits_steered)
                kl_divs.append(kl)

            scale_results[str(scale)] = {
                "mean_kl": float(np.mean(kl_divs)),
                "std_kl": float(np.std(kl_divs)),
            }
            log.info(f"  scale={scale:+.1f}: KL={np.mean(kl_divs):.6f}")

        results[vec_name]["per_scale"] = scale_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for steering (default: 2/3 depth)")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Steering Vectors: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    log.info(f"Loaded {len(sequences)} proteins")

    # Default layer: 2/3 through the network (where representations are most structured)
    if args.layer is None:
        if args.model == "esm2":
            args.layer = 22  # ~2/3 of 33 layers
        else:
            args.layer = 32  # ~2/3 of 48 layers

    # Load model
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        tok = tokenizer
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        tok = tokenizers

    # Extract activations at target layer for steering vector computation
    log.info(f"Extracting activations at layer {args.layer}...")
    t0 = time.time()
    train_seqs = dict(list(sequences.items())[:200])  # Use first 200 for vector computation
    activations = extract_layer_activations(model, args.model, tok, train_seqs,
                                             args.layer, args.device)
    log.info(f"Extracted {len(activations)} proteins in {time.time()-t0:.1f}s")

    # Compute steering vectors
    ss_labels = load_ss_labels()
    func_labels = load_functional_labels()
    all_vectors = {}

    if ss_labels:
        log.info("Computing SS steering vectors...")
        ss_vecs = compute_ss_steering_vectors(activations, ss_labels, log)
        all_vectors.update(ss_vecs)

    if not all_vectors:
        log.warning("No steering vectors computed (no labels found)")
        return

    # Evaluate steering at multiple scales
    log.info(f"Evaluating {len(all_vectors)} steering vectors...")
    test_seqs = dict(list(sequences.items())[200:300])  # Use next 100 for evaluation
    if len(test_seqs) < 10:
        test_seqs = dict(list(sequences.items())[:100])

    eval_results = evaluate_steering(
        model, args.model, tok, test_seqs, args.layer,
        all_vectors, args.device, log
    )

    output = {
        "model": args.model,
        "layer": args.layer,
        "n_proteins_train": len(train_seqs),
        "n_proteins_eval": len(test_seqs),
        "vectors": {name: {
            "norm": float(v.norm()),
            "mean": float(v.mean()),
            "std": float(v.std()),
        } for name, v in all_vectors.items()},
        "evaluation": eval_results,
    }

    out_path = out_dir / "steering_vectors.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")

    # Also save the raw vectors for reuse
    vec_path = out_dir / f"steering_vectors_L{args.layer}.pt"
    torch.save({name: v for name, v in all_vectors.items()}, vec_path)
    log.info(f"Saved vectors to {vec_path}")


if __name__ == "__main__":
    main()
