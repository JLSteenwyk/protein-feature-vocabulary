#!/usr/bin/env python3
"""Causal probing via interchange intervention for ESM-2 and ESM-3.

Tests whether linear probe directions are causally used by the model:
1. Train a linear probe on property P (SS3) at layer L
2. For pairs of residues with different P values, swap the probe's
   readout direction between their activations
3. Measure if model output shifts accordingly

If swapping the SS direction from a helix residue into a sheet residue
changes the model's prediction toward helix-like outputs, then the model
causally reads from that direction — the probe is not epiphenomenal.

Run as:
    ./env/bin/python scripts/unified/run_causal_probing.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_causal_probing.py --model esm3 --device cuda:1

Output: results/unified/{model}/causal_probing.json
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
from sklearn.linear_model import LogisticRegression

from models.metrics import mean_kl_divergence
from models.interventions import (
    cache_residual_stream, interchange_intervention
)


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "causal_probing_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("causal_probe"), out_dir


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


def load_dssp_labels():
    """Load DSSP labels: per-residue SS3 (H=0, E=1, C=2)."""
    dssp_file = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"
    if not dssp_file.exists():
        return None
    with open(dssp_file) as f:
        dssp = json.load(f)
    labels = {}
    ss_map = {'H': 0, 'E': 1, 'C': 2}
    for acc, residues in dssp.items():
        labels[acc] = [ss_map.get(r['ss3'], 2) for r in residues]
    return labels


def train_ss_probe(activations, ss_labels, log):
    """Train a logistic regression probe on SS3 labels.

    Returns the probe and the direction vectors (one per class, binary OvR).
    """
    X_list, y_list = [], []
    for acc, acts in activations.items():
        if acc not in ss_labels:
            continue
        labels = ss_labels[acc]
        L = min(acts.size(0), len(labels))
        X_list.append(acts[:L].numpy())
        y_list.extend(labels[:L])

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    log.info(f"Probe training data: {X.shape[0]} residues, {X.shape[1]} dims")

    probe = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs',
                                multi_class='ovr', n_jobs=-1)
    probe.fit(X, y)
    accuracy = probe.score(X, y)
    log.info(f"Probe accuracy: {accuracy:.4f}")

    # Extract directions: coef_ is (n_classes, n_features)
    directions = {}
    class_names = {0: 'helix', 1: 'sheet', 2: 'coil'}
    for cls_idx in range(probe.coef_.shape[0]):
        d = torch.tensor(probe.coef_[cls_idx], dtype=torch.float32)
        d = d / d.norm()  # Unit normalize
        directions[class_names[cls_idx]] = d

    return probe, directions, accuracy


def extract_activations(model, model_type, tok, sequences, layer, device):
    """Extract activations at a specific layer."""
    activations = {}
    for acc, seq in sequences.items():
        if model_type == "esm2":
            inputs = tok(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad():
                    model(**inputs)
            activations[acc] = cache[layer][0, 1:-1, :].float().cpu()
        else:
            seq_tokens = tok.sequence.encode(seq)
            seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(sequence_tokens=seq_tensor)
            activations[acc] = cache[layer][0, 1:-1, :].float().cpu()
    return activations


def run_interchange_experiment(model, model_type, tok, sequences, ss_labels,
                                layer, directions, device, log, n_pairs=100):
    """Run interchange intervention experiment.

    For each pair (helix-dominant protein, sheet-dominant protein):
    - Cache activations for both
    - Swap the SS direction from source -> target
    - Measure KL divergence and prediction shift
    """
    # Classify proteins by dominant SS
    helix_prots, sheet_prots = [], []
    for acc in sequences:
        if acc not in ss_labels:
            continue
        labels = ss_labels[acc]
        h_frac = sum(1 for l in labels if l == 0) / len(labels)
        e_frac = sum(1 for l in labels if l == 1) / len(labels)
        if h_frac > 0.4:
            helix_prots.append(acc)
        elif e_frac > 0.3:
            sheet_prots.append(acc)

    log.info(f"Helix-dominant: {len(helix_prots)}, Sheet-dominant: {len(sheet_prots)}")
    n_pairs = min(n_pairs, len(helix_prots), len(sheet_prots))

    # Helix->Sheet direction
    direction = directions.get('helix', directions.get('sheet'))
    if direction is None:
        log.error("No helix/sheet direction found")
        return {}

    # Use helix - sheet direction
    if 'helix' in directions and 'sheet' in directions:
        direction = directions['helix'] - directions['sheet']
        direction = direction / direction.norm()

    results = {"helix_to_sheet": [], "sheet_to_helix": []}

    for pair_idx in range(n_pairs):
        h_acc = helix_prots[pair_idx]
        s_acc = sheet_prots[pair_idx]

        # Get baseline logits and activations for both
        if model_type == "esm2":
            h_inputs = tok(sequences[h_acc], return_tensors="pt", truncation=True, max_length=1024).to(device)
            s_inputs = tok(sequences[s_acc], return_tensors="pt", truncation=True, max_length=1024).to(device)

            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad():
                    h_out = model(**h_inputs)
            h_acts = cache[layer]
            h_base_logits = h_out.logits.cpu()

            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad():
                    s_out = model(**s_inputs)
            s_acts = cache[layer]
            s_base_logits = s_out.logits.cpu()

            # Interchange: swap helix direction into sheet protein
            with interchange_intervention(model, model_type, layer, direction, h_acts):
                with torch.no_grad():
                    s_inter_out = model(**s_inputs)
            s_inter_logits = s_inter_out.logits.cpu()

            # And vice versa
            with interchange_intervention(model, model_type, layer, direction, s_acts):
                with torch.no_grad():
                    h_inter_out = model(**h_inputs)
            h_inter_logits = h_inter_out.logits.cpu()

        else:
            h_tokens = tok.sequence.encode(sequences[h_acc])
            s_tokens = tok.sequence.encode(sequences[s_acc])
            h_tensor = torch.tensor(h_tokens, dtype=torch.long, device=device).unsqueeze(0)
            s_tensor = torch.tensor(s_tokens, dtype=torch.long, device=device).unsqueeze(0)

            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    h_out = model(sequence_tokens=h_tensor)
            h_acts = cache[layer]
            h_base_logits = (h_out.sequence_logits if hasattr(h_out, 'sequence_logits')
                             else h_out.logits).float().cpu()

            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s_out = model(sequence_tokens=s_tensor)
            s_acts = cache[layer]
            s_base_logits = (s_out.sequence_logits if hasattr(s_out, 'sequence_logits')
                             else s_out.logits).float().cpu()

            with interchange_intervention(model, model_type, layer, direction, h_acts):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s_inter_out = model(sequence_tokens=s_tensor)
            s_inter_logits = (s_inter_out.sequence_logits if hasattr(s_inter_out, 'sequence_logits')
                              else s_inter_out.logits).float().cpu()

            with interchange_intervention(model, model_type, layer, direction, s_acts):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    h_inter_out = model(sequence_tokens=h_tensor)
            h_inter_logits = (h_inter_out.sequence_logits if hasattr(h_inter_out, 'sequence_logits')
                              else h_inter_out.logits).float().cpu()

        # Measure KL divergence from baseline
        kl_h2s = mean_kl_divergence(s_base_logits, s_inter_logits)
        kl_s2h = mean_kl_divergence(h_base_logits, h_inter_logits)

        results["helix_to_sheet"].append(kl_h2s)
        results["sheet_to_helix"].append(kl_s2h)

        if (pair_idx + 1) % 20 == 0:
            log.info(f"  Pair {pair_idx+1}/{n_pairs}: "
                     f"H→S KL={np.mean(results['helix_to_sheet']):.6f}, "
                     f"S→H KL={np.mean(results['sheet_to_helix']):.6f}")

    # Random direction baseline
    random_kls = []
    rand_dir = torch.randn_like(direction)
    rand_dir = rand_dir / rand_dir.norm()
    for pair_idx in range(min(20, n_pairs)):
        h_acc = helix_prots[pair_idx]
        s_acc = sheet_prots[pair_idx]

        if model_type == "esm2":
            s_inputs = tok(sequences[s_acc], return_tensors="pt", truncation=True, max_length=1024).to(device)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad():
                    model(**tok(sequences[h_acc], return_tensors="pt", truncation=True, max_length=1024).to(device))
            h_acts = cache[layer]
            with torch.no_grad():
                s_base = model(**s_inputs).logits.cpu()
            with interchange_intervention(model, model_type, layer, rand_dir, h_acts):
                with torch.no_grad():
                    s_rand = model(**s_inputs).logits.cpu()
        else:
            h_tokens = tok.sequence.encode(sequences[h_acc])
            s_tokens = tok.sequence.encode(sequences[s_acc])
            h_tensor = torch.tensor(h_tokens, dtype=torch.long, device=device).unsqueeze(0)
            s_tensor = torch.tensor(s_tokens, dtype=torch.long, device=device).unsqueeze(0)
            with cache_residual_stream(model, model_type, [layer]) as cache:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(sequence_tokens=h_tensor)
            h_acts = cache[layer]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                s_base = (model(sequence_tokens=s_tensor).sequence_logits
                          if hasattr(model(sequence_tokens=s_tensor), 'sequence_logits')
                          else model(sequence_tokens=s_tensor).logits).float().cpu()
            with interchange_intervention(model, model_type, layer, rand_dir, h_acts):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    s_rand_out = model(sequence_tokens=s_tensor)
            s_rand = (s_rand_out.sequence_logits if hasattr(s_rand_out, 'sequence_logits')
                      else s_rand_out.logits).float().cpu()

        random_kls.append(mean_kl_divergence(s_base, s_rand))

    results["random_direction_kl"] = random_kls

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer for probing (default: 2/3 depth)")
    parser.add_argument("--n-pairs", type=int, default=100)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Causal Probing: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    ss_labels = load_dssp_labels()
    if ss_labels is None:
        log.error("No DSSP labels found")
        return
    log.info(f"Loaded {len(sequences)} proteins, {len(ss_labels)} with SS labels")

    if args.layer is None:
        args.layer = 22 if args.model == "esm2" else 32

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        tok = tokenizer
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        tok = tokenizers

    # Extract activations for probe training
    log.info(f"Extracting activations at layer {args.layer}...")
    t0 = time.time()
    train_seqs = dict(list(sequences.items())[:200])
    activations = extract_activations(model, args.model, tok, train_seqs,
                                       args.layer, args.device)
    log.info(f"Extracted in {time.time()-t0:.1f}s")

    # Train probe
    log.info("Training SS3 probe...")
    probe, directions, probe_acc = train_ss_probe(activations, ss_labels, log)

    # Run interchange experiment
    log.info(f"Running interchange intervention ({args.n_pairs} pairs)...")
    t0 = time.time()
    test_seqs = dict(list(sequences.items())[200:])
    results = run_interchange_experiment(
        model, args.model, tok, test_seqs, ss_labels,
        args.layer, directions, args.device, log, n_pairs=args.n_pairs
    )
    log.info(f"Interchange done in {time.time()-t0:.1f}s")

    # Summary
    h2s_mean = float(np.mean(results.get("helix_to_sheet", [0])))
    s2h_mean = float(np.mean(results.get("sheet_to_helix", [0])))
    rand_mean = float(np.mean(results.get("random_direction_kl", [0])))
    log.info(f"Results: H→S KL={h2s_mean:.6f}, S→H KL={s2h_mean:.6f}, "
             f"Random KL={rand_mean:.6f}")
    log.info(f"Causal ratio: {(h2s_mean + s2h_mean) / 2 / (rand_mean + 1e-10):.2f}x "
             f"(>1 = probe direction is causal)")

    output = {
        "model": args.model,
        "layer": args.layer,
        "probe_accuracy": probe_acc,
        "n_pairs": args.n_pairs,
        "helix_to_sheet_kl": {
            "mean": h2s_mean,
            "std": float(np.std(results.get("helix_to_sheet", [0]))),
            "values": results.get("helix_to_sheet", []),
        },
        "sheet_to_helix_kl": {
            "mean": s2h_mean,
            "std": float(np.std(results.get("sheet_to_helix", [0]))),
            "values": results.get("sheet_to_helix", []),
        },
        "random_direction_kl": {
            "mean": rand_mean,
            "std": float(np.std(results.get("random_direction_kl", [0]))),
            "values": results.get("random_direction_kl", []),
        },
        "causal_ratio": (h2s_mean + s2h_mean) / 2 / (rand_mean + 1e-10),
        "probe_directions": {name: {"norm": float(d.norm())}
                             for name, d in directions.items()},
    }

    out_path = out_dir / "causal_probing.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
