#!/usr/bin/env python3
"""Analyze WHERE steering effects localize — functional sites vs random positions.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/37_steering_localization.py
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
log = logging.getLogger("steer_loc")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
N_PROTEINS = 100
SEED = 42
ALPHA = 50.0
LAYER = 33

def main():
    from models.esm3_hooks import load_esm3
    from sae.model import build_sae
    from esm.sdk.api import ESMProtein

    with open(STRUCT_DIR / "manifest.json") as f:
        available = json.load(f)["available"]
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)
    meta = json.load(open(EVAL_DIR / "metadata.json"))

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and 50 <= len(sequences[acc]) <= 300 and acc in meta]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()

    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")

    # Build steering vector
    ckpt = torch.load(MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt", map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    W_dec = sae.decoder.weight.detach()  # (d_model, d_sae)
    cm = json.load(open(OUTPUT_DIR / "cross_modal_features.json"))
    enh_ids = cm["enhanced_feature_ids"][:20]
    steer_dir = W_dec[:, enh_ids].mean(dim=1)
    steer_dir = (steer_dir / steer_dir.norm()).cuda()

    # Per-position KL at functional vs non-functional sites
    kl_at_func = []
    kl_at_nonfunc = []
    n_done = 0

    for acc in selected:
        try:
            prot = ESMProtein(sequence=sequences[acc])
            tokens = model.encode(prot)
            seq_tokens = tokens.sequence.to("cuda").unsqueeze(0)
            seq_len = len(sequences[acc])

            # Baseline
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_base = model(sequence_tokens=seq_tokens)
            logits_base = out_base.sequence_logits.float().cpu()[0, 1:-1]  # (L, V)

            # Steered
            def make_hook(direction, strength):
                def hook_fn(module, input, output):
                    return output + strength * direction.to(output.device, output.dtype)
                return hook_fn
            block = model.transformer.blocks[LAYER]
            handle = block.register_forward_hook(make_hook(steer_dir, ALPHA))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_steered = model(sequence_tokens=seq_tokens)
            logits_steered = out_steered.sequence_logits.float().cpu()[0, 1:-1]
            handle.remove()

            # Per-position KL
            log_p = F.log_softmax(logits_base, dim=-1)
            log_q = F.log_softmax(logits_steered, dim=-1)
            p = log_p.exp()
            per_pos_kl = (p * (log_p - log_q)).sum(dim=-1).numpy()  # (L,)

            # Get functional site positions
            func_positions = set()
            for feat in meta[acc].get("features", []):
                ft = feat.get("type", "")
                if ft in ["Active site", "Binding site", "Metal binding", "Site", "Disulfide bond", "Modified residue"]:
                    ps = feat.get("start", 1) - 1
                    pe = feat.get("end", ps + 1)
                    for pos in range(ps, pe):
                        if 0 <= pos < seq_len:
                            func_positions.add(pos)

            for pos in range(min(seq_len, len(per_pos_kl))):
                if pos in func_positions:
                    kl_at_func.append(per_pos_kl[pos])
                else:
                    kl_at_nonfunc.append(per_pos_kl[pos])

            n_done += 1
        except Exception:
            continue
        torch.cuda.empty_cache()

    log.info(f"Processed {n_done} proteins")
    log.info(f"Functional positions: {len(kl_at_func)}, Non-functional: {len(kl_at_nonfunc)}")

    kl_func = np.array(kl_at_func)
    kl_nonfunc = np.array(kl_at_nonfunc)

    log.info(f"\nMean per-position KL at functional sites: {kl_func.mean():.6f}")
    log.info(f"Mean per-position KL at non-functional sites: {kl_nonfunc.mean():.6f}")
    log.info(f"Ratio: {kl_func.mean() / (kl_nonfunc.mean() + 1e-10):.2f}x")

    # What fraction of positions exceed a KL threshold?
    for thresh in [0.001, 0.005, 0.01, 0.05]:
        frac_func = (kl_func > thresh).mean()
        frac_nonfunc = (kl_nonfunc > thresh).mean()
        log.info(f"  KL > {thresh}: functional={frac_func:.3f}, non-functional={frac_nonfunc:.3f}, "
                 f"enrichment={frac_func/(frac_nonfunc+1e-10):.2f}x")

    from scipy import stats
    u, p = stats.mannwhitneyu(kl_func, kl_nonfunc, alternative="greater")
    log.info(f"\nMann-Whitney (func > nonfunc): p={p:.2e}")

    output = {
        "n_proteins": n_done,
        "alpha": ALPHA,
        "n_functional_positions": len(kl_at_func),
        "n_nonfunctional_positions": len(kl_at_nonfunc),
        "mean_kl_functional": float(kl_func.mean()),
        "mean_kl_nonfunctional": float(kl_nonfunc.mean()),
        "ratio": float(kl_func.mean() / (kl_nonfunc.mean() + 1e-10)),
        "mannwhitney_p": float(p),
    }

    out_path = OUTPUT_DIR / "steering_localization.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
