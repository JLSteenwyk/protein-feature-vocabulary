#!/usr/bin/env python3
"""Head ablation under S+St condition for ESM-3.

Complements the S-only head ablation by testing whether head importance
changes when structure tokens are provided.

Focuses on a subset of layers and proteins for efficiency.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/33_head_ablation_sst.py
"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("head_abl_sst")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
N_PROTEINS = 100  # smaller for speed since S+St is slower
SEED = 42
# Test key layers: L0 (geometric attn), L4, L8 (early), L24 (mid), L44, L47 (late)
ABLATION_LAYERS = [0, 4, 8, 24, 44, 47]

def mean_kl(logits_orig, logits_ablated):
    min_L = min(logits_orig.size(1), logits_ablated.size(1))
    log_p = F.log_softmax(logits_orig[:, :min_L].float(), dim=-1)
    log_q = F.log_softmax(logits_ablated[:, :min_L].float(), dim=-1)
    p = log_p.exp()
    return float((p * (log_p - log_q)).sum(dim=-1).mean())

def main():
    from models.esm3_hooks import load_esm3
    from models.interventions import head_ablation
    from esm.sdk.api import ESMProtein

    with open(STRUCT_DIR / "manifest.json") as f:
        available = json.load(f)["available"]
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and 50 <= len(sequences[acc]) <= 400]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins with structures")

    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")
    n_heads = 24
    log.info(f"Testing {len(ABLATION_LAYERS)} layers x {n_heads} heads under S and S+St")

    # Pre-compute baseline logits for both conditions
    log.info("Computing baselines...")
    baselines_s = []
    baselines_sst = []
    valid_proteins = []

    for acc in selected:
        try:
            prot_sst = ESMProtein.from_pdb(available[acc])
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
                out_sst = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)

            baselines_s.append((seq_tokens.cpu(), out_s.sequence_logits.cpu()))
            baselines_sst.append((seq_tokens.cpu(), struct_tokens.cpu(), out_sst.sequence_logits.cpu()))
            valid_proteins.append(acc)
        except Exception as e:
            continue

    log.info(f"  {len(valid_proteins)} proteins with valid baselines")

    # Ablate each head under both conditions
    results_s = {}
    results_sst = {}
    t0 = time.time()

    for layer_idx in ABLATION_LAYERS:
        for head_idx in range(n_heads):
            kl_s_list = []
            kl_sst_list = []

            for pi in range(len(valid_proteins)):
                seq_tokens = baselines_s[pi][0].to("cuda")
                baseline_logits_s = baselines_s[pi][1]

                seq_tokens_sst = baselines_sst[pi][0].to("cuda")
                struct_tokens = baselines_sst[pi][1].to("cuda")
                baseline_logits_sst = baselines_sst[pi][2]

                # S-only ablation
                with head_ablation(model, "esm3", layer_idx, head_idx):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out_abl_s = model(sequence_tokens=seq_tokens)
                kl_s = mean_kl(baseline_logits_s, out_abl_s.sequence_logits.cpu())
                kl_s_list.append(kl_s)

                # S+St ablation
                with head_ablation(model, "esm3", layer_idx, head_idx):
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out_abl_sst = model(sequence_tokens=seq_tokens_sst, structure_tokens=struct_tokens)
                kl_sst = mean_kl(baseline_logits_sst, out_abl_sst.sequence_logits.cpu())
                kl_sst_list.append(kl_sst)

            key = f"L{layer_idx}_H{head_idx}"
            results_s[key] = {"mean_kl": float(np.mean(kl_s_list)), "std_kl": float(np.std(kl_s_list)),
                              "layer": layer_idx, "head": head_idx}
            results_sst[key] = {"mean_kl": float(np.mean(kl_sst_list)), "std_kl": float(np.std(kl_sst_list)),
                                "layer": layer_idx, "head": head_idx}

            log.info(f"  {key}: S={np.mean(kl_s_list):.4f}, S+St={np.mean(kl_sst_list):.4f}, "
                     f"ratio={np.mean(kl_sst_list)/(np.mean(kl_s_list)+1e-10):.2f}")

        elapsed = time.time() - t0
        log.info(f"  Layer {layer_idx} done ({elapsed/60:.1f} min elapsed)")

    # Find top heads under each condition
    top_s = sorted(results_s.items(), key=lambda x: -x[1]["mean_kl"])[:5]
    top_sst = sorted(results_sst.items(), key=lambda x: -x[1]["mean_kl"])[:5]

    top_s_str = [(k, round(v["mean_kl"], 4)) for k, v in top_s]
    top_sst_str = [(k, round(v["mean_kl"], 4)) for k, v in top_sst]
    log.info(f"\nTop 5 heads (S-only): {top_s_str}")
    log.info(f"Top 5 heads (S+St):   {top_sst_str}")

    # L0H7 comparison
    l0h7_s = results_s.get("L0_H7", {}).get("mean_kl", 0)
    l0h7_sst = results_sst.get("L0_H7", {}).get("mean_kl", 0)
    log.info(f"\nL0H7 (geometric attention): S={l0h7_s:.4f}, S+St={l0h7_sst:.4f}, "
             f"ratio={l0h7_sst/(l0h7_s+1e-10):.2f}")

    output = {
        "n_proteins": len(valid_proteins),
        "ablation_layers": ABLATION_LAYERS,
        "n_heads": n_heads,
        "condition_s": {"per_head": results_s},
        "condition_sst": {"per_head": results_sst},
    }

    out_path = OUTPUT_DIR / "head_ablation_sst.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
