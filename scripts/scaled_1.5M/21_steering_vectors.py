#!/usr/bin/env python3
"""Steering vectors: inject/suppress structure-enhanced SAE features.

For structure-enhanced features, compute a steering vector from the SAE decoder,
add/subtract it from ESM-3 residual stream, measure downstream effect on predictions.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/21_steering_vectors.py
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
import h5py

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("steering")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"

N_PROTEINS = 500
N_TOP_FEATURES = 20  # top structure-enhanced features to steer with
STEERING_STRENGTHS = [-50.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 50.0]
LAYER = 33
SEED = 42


def main():
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from models.esm3_hooks import load_esm3, ESM3HookManager
    from esm.sdk.api import ESMProtein
    from sae.model import build_sae

    # Load cross-modal results to get structure-enhanced features
    cm_path = OUTPUT_DIR / "cross_modal_features.json"
    with open(cm_path) as f:
        cm_data = json.load(f)
    enhanced_ids = cm_data["enhanced_feature_ids"][:N_TOP_FEATURES]
    log.info(f"Using top {len(enhanced_ids)} structure-enhanced features for steering")

    # Load SAE to get decoder directions
    sae_path = MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt"
    ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()

    # Compute mean steering vector from decoder columns of enhanced features
    W_dec = sae.decoder.weight.detach()  # (d_model, d_sae)
    steering_vectors = {}
    for fid in enhanced_ids:
        steering_vectors[fid] = W_dec[:, fid].numpy()

    # Mean steering vector across top features
    mean_steer = np.mean([steering_vectors[fid] for fid in enhanced_ids], axis=0)
    mean_steer = mean_steer / np.linalg.norm(mean_steer)  # unit normalize
    mean_steer_tensor = torch.tensor(mean_steer, dtype=torch.float32)
    log.info(f"Mean steering vector norm: {np.linalg.norm(mean_steer):.4f}")

    # Load proteins with structures
    with open(STRUCT_DIR / "manifest.json") as f:
        manifest = json.load(f)
    available = manifest["available"]

    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and len(sequences[acc]) <= 500]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins")

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=args.device)
    log.info("ESM-3 loaded")

    # Also load SAE for feature-level measurement
    from sae.model import build_sae
    from models.esm3_hooks import ESM3HookManager

    sae_ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)
    sae_model = build_sae(sae_ckpt["sae_config"])
    sae_model.load_state_dict(sae_ckpt["model_state_dict"])
    sae_model.eval()

    # Hook manager to capture post-steering activations at layer 33
    hook_mgr = ESM3HookManager(model, layers=[LAYER], extract_attention=False)

    # For each protein: run baseline, then steer at various strengths
    # Measure: (1) KL on logits, (2) target feature activation change, (3) top-1 token change rate
    results = []
    n_done = 0
    start_time = time.time()

    for acc in selected:
        pdb_path = available[acc]

        try:
            protein_s = ESMProtein(sequence=sequences[acc])
            tokens = model.encode(protein_s)
            seq_tokens = tokens.sequence.to(args.device).unsqueeze(0)

            # Baseline: get logits + SAE features
            hook_mgr.cache.clear()
            hook_mgr.register()
            with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                baseline_output = model(sequence_tokens=seq_tokens)
                baseline_logits = baseline_output.sequence_logits.float().cpu()
            baseline_acts = hook_mgr.cache.residual_stream[LAYER][0].float()
            baseline_acts = baseline_acts[1:-1]  # remove BOS/EOS
            hook_mgr.remove()

            with torch.no_grad():
                baseline_z = sae_model.encode(baseline_acts.cpu())
            # Mean activation of target features in baseline
            baseline_target_act = baseline_z[:, enhanced_ids].mean().item()

            protein_result = {"accession": acc, "length": len(sequences[acc]),
                              "baseline_target_feature_act": baseline_target_act}

            for alpha in STEERING_STRENGTHS:
                steer_vec = (mean_steer_tensor * alpha).to(args.device)

                def make_hook(vec):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            return (output[0] + vec.unsqueeze(0).unsqueeze(0),) + output[1:]
                        return output + vec.unsqueeze(0).unsqueeze(0)
                    return hook_fn

                block = model.transformer.blocks[LAYER]
                handle = block.register_forward_hook(make_hook(steer_vec))

                # Also capture post-steering activations
                hook_mgr.cache.clear()
                hook_mgr.register()
                with torch.no_grad(), torch.autocast(args.device, dtype=torch.bfloat16):
                    steered_output = model(sequence_tokens=seq_tokens)
                    steered_logits = steered_output.sequence_logits.float().cpu()
                steered_acts = hook_mgr.cache.residual_stream[LAYER][0].float()
                steered_acts = steered_acts[1:-1]
                hook_mgr.remove()
                handle.remove()

                # SAE feature-level measurement
                with torch.no_grad():
                    steered_z = sae_model.encode(steered_acts.cpu())
                steered_target_act = steered_z[:, enhanced_ids].mean().item()
                target_act_change = steered_target_act - baseline_target_act

                # KL divergence on logits
                baseline_probs = torch.softmax(baseline_logits[0, 1:-1], dim=-1)
                steered_probs = torch.softmax(steered_logits[0, 1:-1], dim=-1)
                kl = torch.sum(baseline_probs * torch.log(baseline_probs / (steered_probs + 1e-10) + 1e-10), dim=-1)

                # Top-1 token change rate
                baseline_top1 = baseline_logits[0, 1:-1].argmax(dim=-1)
                steered_top1 = steered_logits[0, 1:-1].argmax(dim=-1)
                token_change_rate = (baseline_top1 != steered_top1).float().mean().item()

                protein_result[f"alpha_{alpha}"] = {
                    "mean_kl": float(kl.mean().item()),
                    "max_kl": float(kl.max().item()),
                    "target_feature_act": float(steered_target_act),
                    "target_act_change": float(target_act_change),
                    "token_change_rate": float(token_change_rate),
                }

            results.append(protein_result)
            n_done += 1
            torch.cuda.empty_cache()

        except Exception as e:
            if n_done < 5:
                log.warning(f"  Error on {acc}: {e}")
            continue

        if n_done % 50 == 0:
            elapsed = time.time() - start_time
            log.info(f"  {n_done}/{len(selected)} proteins | "
                     f"Rate: {n_done/elapsed:.1f}/s | "
                     f"ETA: {(len(selected)-n_done)/(n_done/elapsed)/60:.0f} min")

    log.info(f"\nProcessed {n_done} proteins")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("STEERING RESULTS")
    log.info(f"{'='*60}")

    summary = {}
    for alpha in STEERING_STRENGTHS:
        entries = [r[f"alpha_{alpha}"] for r in results if f"alpha_{alpha}" in r]
        if entries:
            kls = [e["mean_kl"] for e in entries]
            act_changes = [e["target_act_change"] for e in entries]
            token_changes = [e["token_change_rate"] for e in entries]
            summary[f"alpha_{alpha}"] = {
                "mean_kl": float(np.mean(kls)),
                "median_kl": float(np.median(kls)),
                "mean_target_act_change": float(np.mean(act_changes)),
                "mean_token_change_rate": float(np.mean(token_changes)),
            }
            log.info(f"  alpha={alpha:+5.1f}: KL={np.mean(kls):.4f} | "
                     f"target_act_change={np.mean(act_changes):+.4f} | "
                     f"token_change={100*np.mean(token_changes):.1f}%")

    output = {
        "n_proteins": n_done,
        "n_features_steered": len(enhanced_ids),
        "feature_ids": enhanced_ids,
        "steering_layer": LAYER,
        "strengths": STEERING_STRENGTHS,
        "summary": summary,
        "per_protein": results[:100],  # cap for file size
    }

    out_path = OUTPUT_DIR / "steering_vectors.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
