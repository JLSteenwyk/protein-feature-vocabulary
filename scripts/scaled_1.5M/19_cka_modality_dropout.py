#!/usr/bin/env python3
"""CKA modality dropout: measure representation similarity between S and S+St conditions
across all ESM-3 layers to identify the structure integration zone.

Uses 1000 proteins from the eval set with AlphaFold structures.
Extracts activations at 12 layers under S-only and S+St conditions.
Computes CKA(S, S+St) at each layer.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/19_cka_modality_dropout.py
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cka_dropout")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

# Layers to probe (12 representative layers spanning 0-47)
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 33, 38, 42, 47]
N_PROTEINS = 1000
SEED = 42


def linear_CKA(X, Y):
    """Compute linear CKA between two matrices (n_samples, d)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    hsic_xy = np.linalg.norm(X.T @ Y, 'fro') ** 2
    hsic_xx = np.linalg.norm(X.T @ X, 'fro') ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, 'fro') ** 2

    if hsic_xx * hsic_yy == 0:
        return 0.0
    return hsic_xy / np.sqrt(hsic_xx * hsic_yy)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-proteins", type=int, default=N_PROTEINS)
    args = parser.parse_args()

    from models.esm3_hooks import load_esm3, ESM3HookManager
    from esm.sdk.api import ESMProtein

    # Load structure manifest
    with open(STRUCT_DIR / "manifest.json") as f:
        manifest = json.load(f)
    available = manifest["available"]

    # Sample proteins
    rng = np.random.RandomState(SEED)
    accs_with_struct = sorted(available.keys())
    if len(accs_with_struct) > args.n_proteins:
        selected = rng.choice(accs_with_struct, args.n_proteins, replace=False).tolist()
    else:
        selected = accs_with_struct
    log.info(f"Selected {len(selected)} proteins for CKA analysis")

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=args.device)
    hook_mgr = ESM3HookManager(model, layers=LAYERS, extract_attention=False)
    log.info(f"Extracting {len(LAYERS)} layers: {LAYERS}")

    # Per-layer: accumulate activations for CKA
    # Store mean-pooled representations per protein per layer per condition
    # Only keep proteins where BOTH conditions succeed
    layer_acts_S = {L: [] for L in LAYERS}
    layer_acts_SSt = {L: [] for L in LAYERS}

    n_done = 0
    start_time = time.time()

    for pi, acc in enumerate(selected):
        pdb_path = available[acc]

        try:
            protein_with_struct = ESMProtein.from_pdb(pdb_path)
            if protein_with_struct.sequence is None:
                continue
            pdb_seq = protein_with_struct.sequence
            if len(pdb_seq) < 50 or len(pdb_seq) > 1022:
                continue

            # S-only
            protein_s = ESMProtein(sequence=pdb_seq)
            tokens_s = model.encode(protein_s)

            hook_mgr.cache.clear()
            hook_mgr.register()
            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                model(sequence_tokens=tokens_s.sequence.to(args.device).unsqueeze(0))

            s_results = {}
            for L in LAYERS:
                acts = hook_mgr.cache.residual_stream[L][0].float()
                acts = acts[1:-1]  # remove BOS/EOS
                s_results[L] = acts.mean(dim=0).cpu().numpy()
            hook_mgr.remove()

            # S+St
            tokens_sst = model.encode(protein_with_struct)

            hook_mgr.cache.clear()
            hook_mgr.register()
            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                model(
                    sequence_tokens=tokens_sst.sequence.to(args.device).unsqueeze(0),
                    structure_tokens=tokens_sst.structure.to(args.device).unsqueeze(0),
                )

            sst_results = {}
            for L in LAYERS:
                acts = hook_mgr.cache.residual_stream[L][0].float()
                acts = acts[1:-1]
                sst_results[L] = acts.mean(dim=0).cpu().numpy()
            hook_mgr.remove()

            # Both succeeded — store
            for L in LAYERS:
                layer_acts_S[L].append(s_results[L])
                layer_acts_SSt[L].append(sst_results[L])

            n_done += 1
            torch.cuda.empty_cache()

        except Exception as e:
            if n_done < 5:
                log.warning(f"  Error on {acc}: {e}")
            continue

        if (n_done) % 100 == 0:
            elapsed = time.time() - start_time
            rate = n_done / elapsed
            eta = (len(selected) - n_done) / rate / 60
            log.info(f"  {n_done}/{len(selected)} proteins | Rate: {rate:.1f}/s | ETA: {eta:.0f} min")

    log.info(f"\nExtracted {n_done} proteins in {(time.time()-start_time)/60:.1f} min")

    # Compute CKA at each layer
    log.info("\nComputing CKA(S, S+St) at each layer...")
    cka_results = {}

    for L in LAYERS:
        X_s = np.stack(layer_acts_S[L])    # (n_proteins, d_model)
        X_sst = np.stack(layer_acts_SSt[L])

        cka = linear_CKA(X_s, X_sst)
        cka_results[L] = float(cka)
        log.info(f"  Layer {L:3d}: CKA(S, S+St) = {cka:.4f}")

    # Find integration zone (where CKA starts rising above 0.5)
    layers_sorted = sorted(cka_results.keys())
    integration_start = None
    for L in layers_sorted:
        if cka_results[L] > 0.5 and integration_start is None:
            integration_start = L

    # Find convergence point (CKA > 0.9)
    convergence_point = None
    for L in layers_sorted:
        if cka_results[L] > 0.9:
            convergence_point = L
            break

    log.info(f"\nIntegration zone start (CKA > 0.5): layer {integration_start}")
    log.info(f"Convergence point (CKA > 0.9): layer {convergence_point}")

    output = {
        "n_proteins": n_done,
        "layers": LAYERS,
        "cka_s_vs_sst": cka_results,
        "integration_zone_start": integration_start,
        "convergence_point": convergence_point,
    }

    out_path = OUTPUT_DIR / "cka_modality_dropout.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
