#!/usr/bin/env python3
"""Logit lens: project intermediate ESM-3 representations through the LM head.

At each layer, project the residual stream through the unembedding matrix to see
when predictions form. Compare S-only vs S+St conditions.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/24_logit_lens.py
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
log = logging.getLogger("logit_lens")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

LAYERS = list(range(0, 48, 2))  # every other layer for efficiency
N_PROTEINS = 500
SEED = 42


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from models.esm3_hooks import load_esm3, ESM3HookManager
    from esm.sdk.api import ESMProtein

    # Load structures
    with open(STRUCT_DIR / "manifest.json") as f:
        manifest = json.load(f)
    available = manifest["available"]

    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and len(sequences[acc]) <= 400]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins")

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=args.device)
    hook_mgr = ESM3HookManager(model, layers=LAYERS, extract_attention=False)

    # Get the LM head (unembedding)
    # ESM-3 output heads: model.output_heads.sequence_head
    lm_head = model.output_heads.sequence_head
    # Also need the final layer norm
    final_norm = model.transformer.norm
    log.info("ESM-3 loaded")

    # Per-layer prediction accuracy and KL
    # At each layer: project hidden state → logits → compare with final logits
    layer_accuracy_S = {L: [] for L in LAYERS}
    layer_accuracy_SSt = {L: [] for L in LAYERS}
    layer_kl_S = {L: [] for L in LAYERS}
    layer_kl_SSt = {L: [] for L in LAYERS}

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

            for condition, condition_name, acc_dict, kl_dict in [
                ("S", "S", layer_accuracy_S, layer_kl_S),
                ("S+St", "SSt", layer_accuracy_SSt, layer_kl_SSt),
            ]:
                if condition == "S":
                    protein = ESMProtein(sequence=pdb_seq)
                    tokens = model.encode(protein)
                    input_kwargs = {"sequence_tokens": tokens.sequence.to(args.device).unsqueeze(0)}
                else:
                    tokens = model.encode(protein_with_struct)
                    input_kwargs = {
                        "sequence_tokens": tokens.sequence.to(args.device).unsqueeze(0),
                        "structure_tokens": tokens.structure.to(args.device).unsqueeze(0),
                    }

                hook_mgr.cache.clear()
                hook_mgr.register()
                with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                    output = model(**input_kwargs)
                    final_logits = output.sequence_logits.float().cpu()  # (1, L, vocab)

                final_preds = final_logits[0, 1:-1].argmax(dim=-1)  # remove BOS/EOS
                final_probs = torch.softmax(final_logits[0, 1:-1], dim=-1)

                # Project each layer's hidden state through LM head
                for L in LAYERS:
                    if L not in hook_mgr.cache.residual_stream:
                        continue
                    hidden = hook_mgr.cache.residual_stream[L].to(args.device)
                    # Apply final norm + LM head
                    with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                        normed = final_norm(hidden)
                        layer_logits = lm_head(normed).float().cpu()

                    layer_preds = layer_logits[0, 1:-1].argmax(dim=-1)
                    layer_probs = torch.softmax(layer_logits[0, 1:-1], dim=-1)

                    # Accuracy: fraction of positions matching final prediction
                    accuracy = (layer_preds == final_preds).float().mean().item()
                    acc_dict[L].append(accuracy)

                    # KL divergence from final distribution
                    kl = torch.sum(final_probs * torch.log(final_probs / (layer_probs + 1e-10) + 1e-10), dim=-1)
                    kl_dict[L].append(kl.mean().item())

                hook_mgr.remove()

            n_done += 1
            torch.cuda.empty_cache()

        except Exception as e:
            if n_done < 5:
                log.warning(f"  Error on {acc}: {e}")
            continue

        if n_done % 50 == 0:
            elapsed = time.time() - start_time
            log.info(f"  {n_done}/{len(selected)} | Rate: {n_done/elapsed:.1f}/s | "
                     f"ETA: {(len(selected)-n_done)/(n_done/elapsed)/60:.0f} min")

    log.info(f"\nProcessed {n_done} proteins in {(time.time()-start_time)/60:.1f} min")

    # Compute means
    log.info(f"\n{'='*60}")
    log.info("LOGIT LENS RESULTS")
    log.info(f"{'='*60}")

    results_per_layer = {}
    for L in LAYERS:
        acc_s = np.mean(layer_accuracy_S[L]) if layer_accuracy_S[L] else 0
        acc_sst = np.mean(layer_accuracy_SSt[L]) if layer_accuracy_SSt[L] else 0
        kl_s = np.mean(layer_kl_S[L]) if layer_kl_S[L] else 0
        kl_sst = np.mean(layer_kl_SSt[L]) if layer_kl_SSt[L] else 0

        results_per_layer[L] = {
            "accuracy_S": float(acc_s),
            "accuracy_SSt": float(acc_sst),
            "kl_S": float(kl_s),
            "kl_SSt": float(kl_sst),
            "accuracy_diff": float(acc_sst - acc_s),
            "kl_diff": float(kl_sst - kl_s),
        }

        log.info(f"  Layer {L:3d}: Acc S={acc_s:.3f} SSt={acc_sst:.3f} | "
                 f"KL S={kl_s:.3f} SSt={kl_sst:.3f}")

    output = {
        "n_proteins": n_done,
        "layers": LAYERS,
        "per_layer": results_per_layer,
    }

    out_path = OUTPUT_DIR / "logit_lens.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
