#!/usr/bin/env python3
"""Attribution patching (gradient-based) for ESM-2 and ESM-3.

A fast approximation of activation patching using gradients. Instead of
running N forward passes to patch each layer, computes:

    patching_effect[layer] ≈ grad(loss, hidden[layer]) · (hidden_source[layer] - hidden_target[layer])

One forward + backward pass gives approximate patching effects for ALL layers
simultaneously, making it 10-100× cheaper than full activation patching.

This enables:
1. Fine-grained patching (every layer, not just sampled layers)
2. Per-position patching effects (which positions matter at which layers)
3. Component-level attribution (attention vs MLP contributions)

Uses the same protein pairs as cross-property activation patching
(helix-dominant vs sheet-dominant).

Run as:
    ./env/bin/python scripts/unified/run_attribution_patching.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_attribution_patching.py --model esm3 --device cuda:1

Output: results/unified/{model}/attribution_patching.json
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

from models.interventions import get_layers, cache_residual_stream
from models.metrics import kl_divergence


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "attribution_patching_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("attrpatch"), out_dir


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


def load_dssp_annotations():
    dssp_file = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"
    if dssp_file.exists():
        with open(dssp_file) as f:
            return json.load(f)
    return {}


def classify_protein_ss(dssp_data):
    """Classify protein as helix-dominant, sheet-dominant, or mixed."""
    if not dssp_data:
        return "unknown"
    h_count = sum(1 for r in dssp_data if r.get("ss3") == "H")
    e_count = sum(1 for r in dssp_data if r.get("ss3") == "E")
    total = len(dssp_data)
    if total == 0:
        return "unknown"
    h_frac = h_count / total
    e_frac = e_count / total
    if h_frac > 0.4 and h_frac > e_frac * 2:
        return "helix"
    elif e_frac > 0.2 and e_frac > h_frac * 1.5:
        return "sheet"
    return "mixed"


def forward_with_grad_hooks(model, model_type, inputs, device, all_layers):
    """Run forward pass with gradient-enabled hooks on all layers.

    Returns: (final_logits, per_layer_hidden_states)
    where hidden states have requires_grad=True for backward pass.

    Uses retain_grad() to keep gradients at intermediate layers without
    breaking the computation graph (detach would prevent gradient flow
    to earlier layers).
    """
    layers = get_layers(model, model_type)
    hidden_states = {}
    handles = []

    for layer_idx in all_layers:
        layer = layers[layer_idx]

        def make_hook(lidx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # retain_grad() keeps gradient at this intermediate node
                # without breaking the computation graph
                h.retain_grad()
                hidden_states[lidx] = h
            return hook_fn

        handle = layer.register_forward_hook(make_hook(layer_idx))
        handles.append(handle)

    try:
        if model_type == "esm2":
            out = model(**{k: v.to(device) for k, v in inputs.items()})
            logits = out.logits
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**{k: v.to(device) for k, v in inputs.items()})
            logits = out.sequence_logits if hasattr(out, 'sequence_logits') else out.logits
    finally:
        for h in handles:
            h.remove()

    return logits, hidden_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=500)
    parser.add_argument("--n-pairs", type=int, default=50,
                        help="Number of protein pairs to test")
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Attribution Patching: {args.model} ===")

    sequences = load_sequences(max_proteins=args.max_proteins)
    dssp = load_dssp_annotations()
    log.info(f"Loaded {len(sequences)} proteins, {len(dssp)} with DSSP")

    if args.model == "esm2":
        from models.esm2_hooks import load_esm2
        model, tokenizer = load_esm2(device=args.device)
        n_layers = len(model.esm.encoder.layer)
    else:
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        n_layers = len(model.transformer.blocks)

    log.info(f"{args.model}: {n_layers} layers")
    all_layers = list(range(n_layers))

    # Classify proteins by SS content
    helix_prots = []
    sheet_prots = []
    for acc, seq in sequences.items():
        ss_class = classify_protein_ss(dssp.get(acc, []))
        if ss_class == "helix":
            helix_prots.append((acc, seq))
        elif ss_class == "sheet":
            sheet_prots.append((acc, seq))

    log.info(f"Helix-dominant: {len(helix_prots)}, Sheet-dominant: {len(sheet_prots)}")
    n_pairs = min(args.n_pairs, len(helix_prots), len(sheet_prots))
    log.info(f"Using {n_pairs} pairs")

    # For each pair: compute attribution patching scores
    per_layer_effects = {l: [] for l in all_layers}
    t0 = time.time()

    for pair_idx in range(n_pairs):
        acc_h, seq_h = helix_prots[pair_idx]
        acc_s, seq_s = sheet_prots[pair_idx]

        # Step 1: Cache source (helix) hidden states and logits
        with cache_residual_stream(model, args.model, all_layers) as source_cache:
            if args.model == "esm2":
                inputs_h = tokenizer(seq_h, return_tensors="pt", truncation=True,
                                     max_length=1024).to(args.device)
                with torch.no_grad():
                    out_h = model(**inputs_h)
                source_logits = out_h.logits.detach()
            else:
                seq_tokens_h = tokenizers.sequence.encode(seq_h)
                seq_tensor_h = torch.tensor(seq_tokens_h, dtype=torch.long,
                                            device=args.device).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_h = model(sequence_tokens=seq_tensor_h)
                source_logits = out_h.sequence_logits.detach()

        source_hidden = {l: source_cache[l].to(args.device) for l in all_layers}

        # Step 2: Run target (sheet) with gradient hooks
        if args.model == "esm2":
            inputs_s = tokenizer(seq_s, return_tensors="pt", truncation=True,
                                 max_length=1024)
            target_logits, target_hidden = forward_with_grad_hooks(
                model, args.model, inputs_s, args.device, all_layers)
        else:
            seq_tokens_s = tokenizers.sequence.encode(seq_s)
            seq_tensor_s = torch.tensor(seq_tokens_s, dtype=torch.long).unsqueeze(0)
            target_logits, target_hidden = forward_with_grad_hooks(
                model, args.model, {"sequence_tokens": seq_tensor_s}, args.device, all_layers)

        # Step 3: Compute loss = KL(source || target)
        # Measures how different target is from source
        # Match sequence lengths (source and target proteins differ in length)
        source_log_probs = torch.log_softmax(source_logits.float(), dim=-1).detach()
        source_probs = source_log_probs.exp()
        target_log_probs = torch.log_softmax(target_logits.float(), dim=-1)
        min_L = min(source_log_probs.size(1), target_log_probs.size(1))
        loss = (source_probs[:, :min_L] * (source_log_probs[:, :min_L] - target_log_probs[:, :min_L])).sum()

        # Step 4: Backward to get gradients at each layer
        loss.backward()

        # Step 5: Compute attribution patching score at each layer
        for l in all_layers:
            if l not in target_hidden or target_hidden[l].grad is None:
                continue

            grad = target_hidden[l].grad  # (B, L_target, D)
            src = source_hidden[l]        # (B, L_source, D)
            tgt = target_hidden[l]        # (B, L_target, D)

            # Match sequence lengths
            min_L = min(src.size(1), tgt.size(1))
            diff = src[:, :min_L] - tgt[:, :min_L].detach()

            # Attribution patching score: sum of (grad * diff) over positions and dims
            attr_score = (grad[:, :min_L] * diff).sum().item()
            per_layer_effects[l].append(abs(attr_score))

        # Clean up
        model.zero_grad()
        del target_hidden, source_hidden
        torch.cuda.empty_cache()

        if (pair_idx + 1) % 10 == 0:
            log.info(f"  [{pair_idx+1}/{n_pairs} pairs, {time.time()-t0:.0f}s]")

    log.info(f"Completed {n_pairs} pairs in {time.time()-t0:.1f}s")

    # Aggregate results
    output = {
        "model": args.model,
        "n_layers": n_layers,
        "n_pairs": n_pairs,
        "per_layer": {},
    }

    # Normalize by max for comparison
    all_means = []
    for l in all_layers:
        if per_layer_effects[l]:
            m = float(np.mean(per_layer_effects[l]))
            all_means.append(m)
            output["per_layer"][str(l)] = {
                "mean_effect": m,
                "std_effect": float(np.std(per_layer_effects[l])),
                "median_effect": float(np.median(per_layer_effects[l])),
                "n": len(per_layer_effects[l]),
            }

    # Normalize
    max_effect = max(all_means) if all_means else 1.0
    for l_str in output["per_layer"]:
        output["per_layer"][l_str]["normalized_effect"] = (
            output["per_layer"][l_str]["mean_effect"] / (max_effect + 1e-10)
        )

    # Find peak layer
    if all_means:
        peak_layer = all_layers[int(np.argmax(all_means))]
        output["peak_layer"] = peak_layer
        output["peak_effect"] = float(max(all_means))
        log.info(f"Peak layer: {peak_layer} (effect={max(all_means):.4f})")

    log.info("=== Per-layer attribution patching effects ===")
    for l in all_layers:
        info = output["per_layer"].get(str(l), {})
        norm = info.get("normalized_effect", 0)
        bar = "█" * int(norm * 40)
        log.info(f"  Layer {l:3d}: {bar} ({info.get('mean_effect', 0):.4f})")

    out_path = out_dir / "attribution_patching.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
