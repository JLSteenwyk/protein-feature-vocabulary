#!/usr/bin/env python3
"""Same-protein causal controls: attribution patching + steering on identical proteins under S vs S+St.

Addresses reviewer concern that cross-protein comparisons confound sequence identity with modality effects.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/30_same_protein_causal.py
"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("same_prot")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
N_PROTEINS = 200
ALPHAS = [-50.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 50.0]
LAYER = 33
SEED = 42

def get_layers(model):
    return model.transformer.blocks

def main():
    from models.esm3_hooks import load_esm3
    from esm.sdk.api import ESMProtein
    from sae.model import build_sae

    # Load structures manifest
    with open(STRUCT_DIR / "manifest.json") as f:
        available = json.load(f)["available"]
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys()) if acc in sequences and 50 <= len(sequences[acc]) <= 400]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins with structures")

    # Load ESM-3
    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")
    layers = get_layers(model)
    all_layers = list(range(len(layers)))
    log.info(f"ESM-3 loaded, {len(all_layers)} layers")

    # Load SAE for steering vector
    ckpt = torch.load(MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt", map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    W_dec = sae.decoder.weight.detach()  # (d_model, d_sae)

    cm = json.load(open(OUTPUT_DIR / "cross_modal_features.json"))
    enh_ids = cm["enhanced_feature_ids"][:20]
    steer_dir = W_dec[:, enh_ids].mean(dim=1)
    steer_dir = steer_dir / steer_dir.norm()
    steer_dir = steer_dir.cuda()

    # === Part 1: Same-protein attribution patching ===
    log.info("\n=== Part 1: Same-protein attribution patching ===")
    per_layer_effects = {l: [] for l in all_layers}
    n_done = 0
    t0 = time.time()

    for acc in selected:
        pdb_path = available[acc]
        try:
            prot_sst = ESMProtein.from_pdb(pdb_path)
            if prot_sst.sequence is None: continue
            seq = prot_sst.sequence
            if len(seq) < 50 or len(seq) > 400: continue

            prot_s = ESMProtein(sequence=seq)
            tok_s = model.encode(prot_s)
            tok_sst = model.encode(prot_sst)
            seq_tokens = tok_s.sequence.to("cuda").unsqueeze(0)
            struct_tokens = tok_sst.structure.to("cuda").unsqueeze(0)

            # S-only forward (no grad) — cache hidden states
            s_hidden = {}
            handles = []
            for lidx in all_layers:
                def make_hook(l):
                    def hook_fn(module, input, output):
                        h = output if not isinstance(output, tuple) else output[0]
                        s_hidden[l] = h.detach()
                    return hook_fn
                handles.append(layers[lidx].register_forward_hook(make_hook(lidx)))

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_s = model(sequence_tokens=seq_tokens)
                logits_s = out_s.sequence_logits.float()
            for h in handles: h.remove()

            # S+St forward (with grad) — retain_grad on hidden states
            sst_hidden = {}
            handles = []
            for lidx in all_layers:
                def make_hook2(l):
                    def hook_fn(module, input, output):
                        h = output if not isinstance(output, tuple) else output[0]
                        h.retain_grad()
                        sst_hidden[l] = h
                    return hook_fn
                handles.append(layers[lidx].register_forward_hook(make_hook2(lidx)))

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_sst = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
                logits_sst = out_sst.sequence_logits.float()
            for h in handles: h.remove()

            # Loss = KL(S || S+St) — how different is S+St from S?
            min_L = min(logits_s.size(1), logits_sst.size(1))
            s_log_probs = torch.log_softmax(logits_s[:, :min_L].detach(), dim=-1)
            s_probs = s_log_probs.exp()
            sst_log_probs = torch.log_softmax(logits_sst[:, :min_L], dim=-1)
            loss = (s_probs * (s_log_probs - sst_log_probs)).sum()
            loss.backward()

            # Attribution at each layer
            for l in all_layers:
                if l in sst_hidden and sst_hidden[l].grad is not None and l in s_hidden:
                    grad = sst_hidden[l].grad
                    diff = sst_hidden[l].detach()[:, :min_L] - s_hidden[l][:, :min_L].to(grad.device, grad.dtype)
                    attr = (grad[:, :min_L] * diff).sum().abs().item()
                    per_layer_effects[l].append(attr)

            model.zero_grad()
            n_done += 1

        except Exception as e:
            if n_done < 3: log.warning(f"Error {acc}: {e}")
            continue

        if n_done % 50 == 0:
            log.info(f"  {n_done}/{len(selected)} | {time.time()-t0:.0f}s")
        torch.cuda.empty_cache()

    log.info(f"Attribution done: {n_done} proteins in {(time.time()-t0)/60:.1f} min")

    # === Part 2: Same-protein steering validation ===
    log.info("\n=== Part 2: Same-protein steering validation ===")
    steering_results = {f"alpha_{a}": [] for a in ALPHAS}
    baseline_kls = []
    n_steer = 0

    for acc in selected[:N_PROTEINS]:
        pdb_path = available[acc]
        try:
            prot_sst = ESMProtein.from_pdb(pdb_path)
            if prot_sst.sequence is None: continue
            seq = prot_sst.sequence
            if len(seq) < 50 or len(seq) > 400: continue

            prot_s = ESMProtein(sequence=seq)
            tok_s = model.encode(prot_s)
            tok_sst = model.encode(prot_sst)
            seq_tokens = tok_s.sequence.to("cuda").unsqueeze(0)
            struct_tokens = tok_sst.structure.to("cuda").unsqueeze(0)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_s = model(sequence_tokens=seq_tokens)
                logits_s = out_s.sequence_logits.float().cpu()[0, 1:-1]

                out_sst = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
                logits_sst = out_sst.sequence_logits.float().cpu()[0, 1:-1]

            min_L = min(logits_s.size(0), logits_sst.size(0))
            probs_s = torch.softmax(logits_s[:min_L], dim=-1)
            probs_sst = torch.softmax(logits_sst[:min_L], dim=-1)
            baseline_kl = torch.sum(probs_s * torch.log(probs_s / (probs_sst + 1e-10) + 1e-10), dim=-1).mean().item()
            baseline_kls.append(baseline_kl)

            for alpha in ALPHAS:
                def make_hook(direction, strength):
                    def hook_fn(module, input, output):
                        return output + strength * direction.to(output.device, output.dtype)
                    return hook_fn

                block = model.transformer.blocks[LAYER]
                handle = block.register_forward_hook(make_hook(steer_dir, alpha))
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_steered = model(sequence_tokens=seq_tokens)
                    logits_steered = out_steered.sequence_logits.float().cpu()[0, 1:-1]
                handle.remove()

                probs_steered = torch.softmax(logits_steered[:min_L], dim=-1)
                kl_to_sst = torch.sum(probs_steered * torch.log(probs_steered / (probs_sst + 1e-10) + 1e-10), dim=-1).mean().item()
                steering_results[f"alpha_{alpha}"].append({
                    "kl_to_sst": kl_to_sst,
                    "baseline_kl": baseline_kl,
                    "ratio": kl_to_sst / (baseline_kl + 1e-10),
                })

            n_steer += 1
        except Exception as e:
            if n_steer < 3: log.warning(f"Steering error {acc}: {e}")
            continue

        if n_steer % 50 == 0:
            log.info(f"  Steering: {n_steer}/{len(selected)}")
        torch.cuda.empty_cache()

    log.info(f"Steering done: {n_steer} proteins")

    # Aggregate
    attr_results = {}
    for l in all_layers:
        vals = per_layer_effects[l]
        attr_results[l] = {
            "mean_effect": float(np.mean(vals)) if vals else 0,
            "std_effect": float(np.std(vals)) if vals else 0,
            "n": len(vals),
        }

    peak_layer = max(attr_results.keys(), key=lambda l: attr_results[l]["mean_effect"])
    log.info(f"\nAttribution peak: L{peak_layer} (effect={attr_results[peak_layer]['mean_effect']:.1f})")

    steer_agg = {}
    for akey, vals in steering_results.items():
        if vals:
            mean_kl_to_sst = np.mean([v["kl_to_sst"] for v in vals])
            mean_baseline = np.mean([v["baseline_kl"] for v in vals])
            mean_ratio = np.mean([v["ratio"] for v in vals])
            reduces = np.mean([v["kl_to_sst"] < v["baseline_kl"] for v in vals])
            steer_agg[akey] = {
                "mean_kl_to_sst": float(mean_kl_to_sst),
                "mean_baseline_kl": float(mean_baseline),
                "mean_ratio": float(mean_ratio),
                "frac_reducing_distance": float(reduces),
            }
            log.info(f"  {akey}: KL_to_sst={mean_kl_to_sst:.4f}, baseline={mean_baseline:.4f}, ratio={mean_ratio:.3f}, reduces={reduces:.2f}")

    output = {
        "n_proteins_attribution": n_done,
        "n_proteins_steering": n_steer,
        "same_protein_attribution": {
            "per_layer": attr_results,
            "peak_layer": int(peak_layer),
            "peak_effect": float(attr_results[peak_layer]["mean_effect"]),
        },
        "same_protein_steering": {
            "steering_layer": LAYER,
            "n_features": len(enh_ids),
            "per_alpha": steer_agg,
            "mean_baseline_kl": float(np.mean(baseline_kls)) if baseline_kls else 0,
        },
    }

    out_path = OUTPUT_DIR / "same_protein_causal.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
