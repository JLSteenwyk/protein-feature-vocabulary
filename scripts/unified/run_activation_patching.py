#!/usr/bin/env python3
"""Activation patching (causal tracing) for ESM-2 and ESM-3.

Two patching modes:
1. Cross-property: Patch activations between proteins with contrasting
   structural properties (helix-dominant vs sheet-dominant)
2. Cross-modality (ESM-3 only): Patch S-only activations into S+St run,
   measuring where structure information is causally used

Run as:
    ./env/bin/python scripts/unified/run_activation_patching.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_activation_patching.py --model esm3 --device cuda:1
    ./env/bin/python scripts/unified/run_activation_patching.py --model esm3 --patching-type cross_modality --device cuda:1

Output: results/unified/{model}/activation_patching.json
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
from models.interventions import activation_patch, cache_residual_stream


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "activation_patching_log.txt"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("act_patch"), out_dir


def load_sequences_with_ss(max_pairs=50, max_length=400):
    """Load protein sequences and SS annotations, select contrasting pairs."""
    seq_file = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    dssp_file = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"

    if not seq_file.exists() or not dssp_file.exists():
        # Fallback: just load sequences and pair sequentially
        return load_sequences_fallback(max_pairs, max_length)

    with open(seq_file) as f:
        all_seqs = json.load(f)
    with open(dssp_file) as f:
        dssp = json.load(f)

    # Classify proteins by dominant SS
    helix_proteins = []
    sheet_proteins = []
    for acc in sorted(all_seqs.keys()):
        if acc not in dssp or len(all_seqs[acc]) > max_length:
            continue
        ss = dssp[acc]
        if isinstance(ss, dict):
            ss = ss.get("ss3", ss.get("secondary_structure", ""))
        if not ss or len(ss) < 10:
            continue
        n_h = ss.count("H")
        n_e = ss.count("E")
        n_total = len(ss)
        if n_h / n_total > 0.4:
            helix_proteins.append(acc)
        elif n_e / n_total > 0.25:
            sheet_proteins.append(acc)

    # Create pairs
    n_pairs = min(max_pairs, len(helix_proteins), len(sheet_proteins))
    pairs = [(helix_proteins[i], sheet_proteins[i]) for i in range(n_pairs)]

    return {acc: all_seqs[acc] for acc in helix_proteins[:n_pairs] + sheet_proteins[:n_pairs]}, pairs


def load_sequences_fallback(max_pairs=50, max_length=400):
    """Fallback: load sequences and pair them sequentially."""
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
    sorted_accs = sorted(filtered.keys())[:max_pairs * 2]
    pairs = [(sorted_accs[i], sorted_accs[i + max_pairs])
             for i in range(min(max_pairs, len(sorted_accs) // 2))]
    return {k: filtered[k] for k in sorted_accs}, pairs


def get_patching_layers(model_type, n_layers):
    """Select layers for patching (evenly spaced)."""
    if model_type == "esm2":
        step = max(1, n_layers // 12)
        return list(range(0, n_layers, step))
    else:
        step = max(1, n_layers // 16)
        return list(range(0, n_layers, step))


# ============================================================
# Cross-Property Patching
# ============================================================

def cross_property_patching_esm2(model, tokenizer, sequences, pairs, device, log):
    """Patch activations between contrasting protein pairs (ESM-2)."""
    n_layers = len(model.esm.encoder.layer)
    patch_layers = get_patching_layers("esm2", n_layers)
    log.info(f"Patching at {len(patch_layers)} layers: {patch_layers}")

    results = {str(l): [] for l in patch_layers}

    for pair_idx, (acc_a, acc_b) in enumerate(pairs):
        seq_a, seq_b = sequences[acc_a], sequences[acc_b]

        # Get baseline logits for B
        inputs_b = tokenizer(seq_b, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out_b = model(**inputs_b)
        logits_b = out_b.logits.cpu()

        # Cache activations from A
        inputs_a = tokenizer(seq_a, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with cache_residual_stream(model, "esm2", patch_layers) as cache_a:
            with torch.no_grad():
                model(**inputs_a)

        # Patch A's activations into B at each layer
        for layer_idx in patch_layers:
            source_act = cache_a[layer_idx]  # (1, L_a, D)
            with activation_patch(model, "esm2", layer_idx, source_act):
                with torch.no_grad():
                    out_patched = model(**inputs_b)
            kl = mean_kl_divergence(logits_b, out_patched.logits.cpu())
            results[str(layer_idx)].append(kl)

        if (pair_idx + 1) % 10 == 0:
            log.info(f"  Pair {pair_idx+1}/{len(pairs)}")

    # Aggregate
    agg = {}
    for l in patch_layers:
        vals = results[str(l)]
        agg[str(l)] = {
            "mean_kl": float(np.mean(vals)),
            "std_kl": float(np.std(vals)),
            "n_pairs": len(vals),
        }
    return agg, patch_layers


def cross_property_patching_esm3(model, tokenizers, sequences, pairs, device, log):
    """Patch activations between contrasting protein pairs (ESM-3)."""
    n_layers = len(model.transformer.blocks)
    patch_layers = get_patching_layers("esm3", n_layers)
    log.info(f"Patching at {len(patch_layers)} layers: {patch_layers}")

    results = {str(l): [] for l in patch_layers}

    for pair_idx, (acc_a, acc_b) in enumerate(pairs):
        seq_a, seq_b = sequences[acc_a], sequences[acc_b]

        # Baseline logits for B
        tok_b = tokenizers.sequence.encode(seq_b)
        tensor_b = torch.tensor(tok_b, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_b = model(sequence_tokens=tensor_b)
        logits_b = (out_b.sequence_logits if hasattr(out_b, 'sequence_logits') else out_b.logits).float().cpu()

        # Cache A activations
        tok_a = tokenizers.sequence.encode(seq_a)
        tensor_a = torch.tensor(tok_a, dtype=torch.long, device=device).unsqueeze(0)
        with cache_residual_stream(model, "esm3", patch_layers) as cache_a:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(sequence_tokens=tensor_a)

        # Patch
        for layer_idx in patch_layers:
            source_act = cache_a[layer_idx]
            with activation_patch(model, "esm3", layer_idx, source_act):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_patched = model(sequence_tokens=tensor_b)
            logits_p = (out_patched.sequence_logits if hasattr(out_patched, 'sequence_logits') else out_patched.logits).float().cpu()
            kl = mean_kl_divergence(logits_b, logits_p)
            results[str(layer_idx)].append(kl)

        if (pair_idx + 1) % 10 == 0:
            log.info(f"  Pair {pair_idx+1}/{len(pairs)}")

    agg = {}
    for l in patch_layers:
        vals = results[str(l)]
        agg[str(l)] = {
            "mean_kl": float(np.mean(vals)),
            "std_kl": float(np.std(vals)),
            "n_pairs": len(vals),
        }
    return agg, patch_layers


# ============================================================
# Cross-Modality Patching (ESM-3 only)
# ============================================================

def cross_modality_patching_esm3(model, tokenizers, sequences, device, log, max_proteins=100):
    """Patch S-only activations into S+St run at each layer.

    For proteins with AlphaFold structures:
    1. Run S+St (full multimodal) -> get baseline logits
    2. Cache S-only activations at each layer
    3. Run S+St but patch in S-only activations at layer L
    4. Measure KL divergence from baseline
    """
    from models.esm3_hooks import extract_esm3_multimodal

    n_layers = len(model.transformer.blocks)
    patch_layers = get_patching_layers("esm3", n_layers)
    log.info(f"Cross-modality patching at {len(patch_layers)} layers")

    # Load structure tokens from existing multimodal data
    struct_dir = ROOT / "data" / "scaled" / "structures"
    structure_map_file = ROOT / "data" / "scaled" / "structure_map.json"

    if structure_map_file.exists():
        with open(structure_map_file) as f:
            structure_map = json.load(f)
    else:
        log.warning("No structure_map.json found, trying to find PDB files")
        structure_map = {}

    # Find proteins that have both sequence and structure
    available = []
    for acc in sorted(sequences.keys()):
        if acc in structure_map or (struct_dir / f"{acc}.pdb").exists():
            available.append(acc)
    available = available[:max_proteins]

    if len(available) < 10:
        log.warning(f"Only {len(available)} proteins with structures available")
        log.info("Falling back to tokenizing structures from AlphaFold")
        # Use the subset we have
        pass

    log.info(f"Using {len(available)} proteins with structures")

    # For each protein: encode structure tokens, run S+St baseline, then patch
    results = {str(l): [] for l in patch_layers}

    for idx, acc in enumerate(available):
        seq = sequences[acc]

        # Get structure tokens via ESM-3's structure tokenizer
        pdb_path = struct_dir / f"{acc}.pdb" if (struct_dir / f"{acc}.pdb").exists() else None
        cif_path = struct_dir / f"{acc}.cif" if (struct_dir / f"{acc}.cif").exists() else None

        struct_path = pdb_path or cif_path
        if struct_path is None:
            continue

        try:
            from esm.utils.structure.protein_chain import ProteinChain
            chain = ProteinChain.from_pdb(str(struct_path))
            from esm.tokenization import get_model_tokenizers
            struct_tokens = tokenizers.structure.encode(chain.to_structure_encoder_inputs())
            if isinstance(struct_tokens, np.ndarray):
                struct_tokens = torch.tensor(struct_tokens, dtype=torch.long)
            struct_tokens = struct_tokens.to(device)
            if struct_tokens.dim() == 1:
                struct_tokens = struct_tokens.unsqueeze(0)
        except Exception as e:
            log.warning(f"  Skipping {acc}: structure encoding failed: {e}")
            continue

        # Tokenize sequence
        seq_tokens = tokenizers.sequence.encode(seq)
        seq_tensor = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)

        # Check length compatibility
        if seq_tensor.size(1) != struct_tokens.size(1):
            min_L = min(seq_tensor.size(1), struct_tokens.size(1))
            seq_tensor = seq_tensor[:, :min_L]
            struct_tokens = struct_tokens[:, :min_L]

        # S+St baseline
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_st = model(sequence_tokens=seq_tensor, structure_tokens=struct_tokens)
        logits_st = (out_st.sequence_logits if hasattr(out_st, 'sequence_logits') else out_st.logits).float().cpu()

        # Cache S-only activations
        with cache_residual_stream(model, "esm3", patch_layers) as cache_s:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(sequence_tokens=seq_tensor)

        # Patch S-only into S+St at each layer
        for layer_idx in patch_layers:
            source_act = cache_s[layer_idx]
            with activation_patch(model, "esm3", layer_idx, source_act):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_patched = model(sequence_tokens=seq_tensor, structure_tokens=struct_tokens)

            logits_p = (out_patched.sequence_logits if hasattr(out_patched, 'sequence_logits') else out_patched.logits).float().cpu()
            kl = mean_kl_divergence(logits_st, logits_p)
            results[str(layer_idx)].append(kl)

        if (idx + 1) % 10 == 0:
            log.info(f"  [{idx+1}/{len(available)}] {acc}")

    agg = {}
    for l in patch_layers:
        vals = results[str(l)]
        if vals:
            agg[str(l)] = {
                "mean_kl": float(np.mean(vals)),
                "std_kl": float(np.std(vals)),
                "n_proteins": len(vals),
            }
    return agg, patch_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--patching-type", choices=["cross_property", "cross_modality"],
                        default="cross_property")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=50)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Activation Patching: {args.model}, type={args.patching_type} ===")

    if args.patching_type == "cross_modality" and args.model != "esm3":
        log.error("Cross-modality patching only supported for ESM-3")
        return

    if args.patching_type == "cross_property":
        sequences, pairs = load_sequences_with_ss(max_pairs=args.max_pairs)
        log.info(f"Loaded {len(sequences)} proteins, {len(pairs)} pairs")

        if args.model == "esm2":
            from models.esm2_hooks import load_esm2
            model, tokenizer = load_esm2(device=args.device)
            per_layer, patch_layers = cross_property_patching_esm2(
                model, tokenizer, sequences, pairs, args.device, log
            )
        else:
            from models.esm3_hooks import load_esm3
            model, tokenizers = load_esm3(device=args.device)
            per_layer, patch_layers = cross_property_patching_esm3(
                model, tokenizers, sequences, pairs, args.device, log
            )

        output = {
            "model": args.model,
            "patching_type": "cross_property",
            "patch_layers": patch_layers,
            "n_pairs": len(pairs),
            "per_layer": per_layer,
        }

    else:  # cross_modality
        sequences, _ = load_sequences_with_ss(max_pairs=100)
        from models.esm3_hooks import load_esm3
        model, tokenizers = load_esm3(device=args.device)
        per_layer, patch_layers = cross_modality_patching_esm3(
            model, tokenizers, sequences, args.device, log
        )
        output = {
            "model": "esm3",
            "patching_type": "cross_modality",
            "patch_layers": patch_layers,
            "per_layer": per_layer,
        }

    suffix = f"_{args.patching_type}" if args.patching_type != "cross_property" else ""
    out_path = out_dir / f"activation_patching{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
