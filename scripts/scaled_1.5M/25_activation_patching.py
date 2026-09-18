#!/usr/bin/env python3
"""Activation patching: identify which layers carry structure information in ESM-3.

Uses additive patching: at each layer, adds the modality difference
(S+St hidden - S hidden) to the S-only run, measuring how much each layer's
structure contribution shifts output toward S+St predictions.

Also measures denoising (removing structure contribution from S+St run).

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/25_activation_patching.py
"""

import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("act_patch")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

PATCH_LAYERS = list(range(0, 48, 3))  # every 3rd layer
N_PROTEINS = 200
SEED = 42


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from models.esm3_hooks import load_esm3, ESM3HookManager
    from esm.sdk.api import ESMProtein

    with open(STRUCT_DIR / "manifest.json") as f:
        manifest = json.load(f)
    available = manifest["available"]

    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and len(sequences[acc]) <= 400]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins for activation patching")

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=args.device)
    log.info("ESM-3 loaded")

    # Results per layer
    # "inject": add (S+St - S) difference at layer L to S-only run → how much toward S+St?
    # "remove": subtract (S+St - S) difference at layer L from S+St run → how much away from S+St?
    layer_kl_inject = {L: [] for L in PATCH_LAYERS}
    layer_kl_remove = {L: [] for L in PATCH_LAYERS}
    # Track recovery fraction: what fraction of KL(S||S+St) does patching at L recover?
    layer_recovery = {L: [] for L in PATCH_LAYERS}

    n_done = 0
    start_time = time.time()

    for acc in selected:
        pdb_path = available[acc]

        try:
            protein_with_struct = ESMProtein.from_pdb(pdb_path)
            if protein_with_struct.sequence is None:
                continue
            pdb_seq = protein_with_struct.sequence
            if len(pdb_seq) < 50 or len(pdb_seq) > 400:
                continue

            protein_s = ESMProtein(sequence=pdb_seq)
            tokens_s = model.encode(protein_s)
            tokens_sst = model.encode(protein_with_struct)

            seq_tokens = tokens_s.sequence.to(args.device).unsqueeze(0)
            struct_tokens = tokens_sst.structure.to(args.device).unsqueeze(0)

            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                # Step 1: Cache per-layer outputs for S-only using hook manager
                mgr = ESM3HookManager(model, layers=PATCH_LAYERS)
                mgr.register()
                out_s = model(sequence_tokens=seq_tokens)
                logits_s = out_s.sequence_logits.float().cpu()[0, 1:-1]
                h_s = {L: mgr.cache.residual_stream[L].clone() for L in PATCH_LAYERS}
                mgr.remove()

                # Step 2: Cache per-layer outputs for S+St
                mgr.register()
                out_sst = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
                logits_sst = out_sst.sequence_logits.float().cpu()[0, 1:-1]
                h_sst = {L: mgr.cache.residual_stream[L].clone() for L in PATCH_LAYERS}
                mgr.remove()

                probs_s = torch.softmax(logits_s, dim=-1)
                probs_sst = torch.softmax(logits_sst, dim=-1)
                baseline_kl = torch.sum(
                    probs_s * torch.log(probs_s / (probs_sst + 1e-10) + 1e-10), dim=-1
                ).mean().item()

                # Step 3: For each layer, additive patching
                for patch_L in PATCH_LAYERS:
                    # Compute modality difference at this layer
                    delta = (h_sst[patch_L] - h_s[patch_L]).to(args.device)

                    # INJECT: add delta to S-only run at layer L
                    # Hook on block[L] output: add delta
                    def make_add_hook(d):
                        def hook_fn(module, input, output):
                            return output + d.to(output.device, output.dtype)
                        return hook_fn

                    def make_sub_hook(d):
                        def hook_fn(module, input, output):
                            return output - d.to(output.device, output.dtype)
                        return hook_fn

                    # Inject structure difference into S-only run
                    block = model.transformer.blocks[patch_L]
                    handle = block.register_forward_hook(make_add_hook(delta))
                    out_inj = model(sequence_tokens=seq_tokens)
                    inj_logits = out_inj.sequence_logits.float().cpu()[0, 1:-1]
                    handle.remove()

                    inj_probs = torch.softmax(inj_logits, dim=-1)
                    # KL(S || injected): how much did output change from S?
                    kl_inj = torch.sum(
                        probs_s * torch.log(probs_s / (inj_probs + 1e-10) + 1e-10), dim=-1
                    ).mean().item()
                    layer_kl_inject[patch_L].append(kl_inj)

                    # Recovery: KL(injected || S+St) / KL(S || S+St)
                    # Lower means patching at L recovered more of the S+St signal
                    kl_inj_to_sst = torch.sum(
                        inj_probs * torch.log(inj_probs / (probs_sst + 1e-10) + 1e-10), dim=-1
                    ).mean().item()
                    recovery = 1.0 - (kl_inj_to_sst / (baseline_kl + 1e-10))
                    layer_recovery[patch_L].append(max(0, min(1, recovery)))

                    # Remove structure difference from S+St run
                    handle = block.register_forward_hook(make_sub_hook(delta))
                    out_rem = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
                    rem_logits = out_rem.sequence_logits.float().cpu()[0, 1:-1]
                    handle.remove()

                    rem_probs = torch.softmax(rem_logits, dim=-1)
                    kl_rem = torch.sum(
                        probs_sst * torch.log(probs_sst / (rem_probs + 1e-10) + 1e-10), dim=-1
                    ).mean().item()
                    layer_kl_remove[patch_L].append(kl_rem)

            n_done += 1
            torch.cuda.empty_cache()

        except Exception as e:
            if n_done < 5:
                log.warning(f"  Error on {acc}: {e}")
                import traceback
                traceback.print_exc()
            continue

        if n_done % 20 == 0:
            elapsed = time.time() - start_time
            rate = n_done / elapsed
            log.info(f"  {n_done}/{len(selected)} | Rate: {rate:.2f}/s")
            if layer_kl_inject[0]:
                l0_inj = np.mean(layer_kl_inject[0])
                l24_inj = np.mean(layer_kl_inject[24])
                l45_inj = np.mean(layer_kl_inject[45])
                l0_rec = np.mean(layer_recovery[0])
                l45_rec = np.mean(layer_recovery[45])
                log.info(f"    inject KL: L0={l0_inj:.4f}, L24={l24_inj:.4f}, L45={l45_inj:.4f}")
                log.info(f"    recovery: L0={l0_rec:.3f}, L45={l45_rec:.3f}")

    log.info(f"\nProcessed {n_done} proteins in {(time.time()-start_time)/60:.1f} min")

    # Results
    log.info(f"\n{'='*60}")
    log.info("ACTIVATION PATCHING RESULTS (ADDITIVE)")
    log.info(f"{'='*60}")

    results_per_layer = {}
    for L in PATCH_LAYERS:
        kl_inj = np.mean(layer_kl_inject[L]) if layer_kl_inject[L] else 0
        kl_rem = np.mean(layer_kl_remove[L]) if layer_kl_remove[L] else 0
        rec = np.mean(layer_recovery[L]) if layer_recovery[L] else 0
        results_per_layer[L] = {
            "kl_inject_structure": float(kl_inj),
            "kl_remove_structure": float(kl_rem),
            "recovery_fraction": float(rec),
            "n_proteins": len(layer_kl_inject[L]),
        }
        log.info(f"  Layer {L:3d}: inject={kl_inj:.4f}, remove={kl_rem:.4f}, recovery={rec:.3f}")

    output = {
        "n_proteins": n_done,
        "patch_layers": PATCH_LAYERS,
        "method": "additive",
        "per_layer": results_per_layer,
    }

    out_path = OUTPUT_DIR / "activation_patching.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
