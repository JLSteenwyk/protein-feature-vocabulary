#!/usr/bin/env python3
"""Steering vector controls: real vs random vs shuffled vs orthogonalized directions.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/31_steering_controls.py
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
log = logging.getLogger("steer_ctrl")

EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
N_PROTEINS = 200
ALPHAS = [-50.0, -20.0, -10.0, -5.0, 5.0, 10.0, 20.0, 50.0]
LAYER = 33
SEED = 42

def main():
    from models.esm3_hooks import load_esm3, ESM3HookManager
    from esm.sdk.api import ESMProtein
    from sae.model import build_sae

    # Load model
    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")
    log.info("ESM-3 loaded")

    # Load SAE for decoder weights
    ckpt = torch.load(MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt", map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    W_dec = sae.decoder.weight.detach().numpy()  # (d_model, d_sae)

    # Enhanced feature IDs
    cm = json.load(open(OUTPUT_DIR / "cross_modal_features.json"))
    enh_ids = cm["enhanced_feature_ids"][:20]

    # Real steering vector: mean decoder column of enhanced features
    real_dir = W_dec[:, enh_ids].mean(axis=1)
    real_dir = real_dir / np.linalg.norm(real_dir)

    # Control directions
    rng = np.random.RandomState(SEED)
    random_dir = rng.randn(W_dec.shape[0]).astype(np.float32)
    random_dir /= np.linalg.norm(random_dir)

    # Shuffled decoder: average of 20 random NON-enhanced features
    all_ids = list(range(W_dec.shape[1]))
    non_enh = [i for i in all_ids if i not in set(enh_ids)]
    shuf_ids = rng.choice(non_enh, 20, replace=False)
    shuf_dir = W_dec[:, shuf_ids].mean(axis=1)
    shuf_dir = shuf_dir / np.linalg.norm(shuf_dir)

    # Orthogonalized: random minus projection onto real
    orth_dir = random_dir - np.dot(random_dir, real_dir) * real_dir
    orth_dir = orth_dir / np.linalg.norm(orth_dir)

    directions = {
        "real": torch.tensor(real_dir, dtype=torch.float32).cuda(),
        "random": torch.tensor(random_dir, dtype=torch.float32).cuda(),
        "shuffled_decoder": torch.tensor(shuf_dir, dtype=torch.float32).cuda(),
        "orthogonalized": torch.tensor(orth_dir, dtype=torch.float32).cuda(),
    }

    # Load proteins
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)
    prots = [(acc, seq) for acc, seq in sorted(sequences.items()) if 50 <= len(seq) <= 500]
    rng2 = np.random.RandomState(SEED)
    sel = rng2.choice(len(prots), min(N_PROTEINS, len(prots)), replace=False)
    selected = [prots[i] for i in sel]
    log.info(f"Selected {len(selected)} proteins")

    # Encode enhanced features for activation measurement
    enh_enc_cols = sae.encoder.weight[enh_ids].detach().cuda()  # (20, d_model)

    results = {d: {f"alpha_{a}": [] for a in ALPHAS} for d in directions}
    t0 = time.time()

    for pi, (acc, seq) in enumerate(selected):
        try:
            protein = ESMProtein(sequence=seq)
            tokens = model.encode(protein)
            seq_tokens = tokens.sequence.to("cuda").unsqueeze(0)

            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_base = model(sequence_tokens=seq_tokens)
                logits_base = out_base.sequence_logits.float().cpu()[0, 1:-1]
            probs_base = torch.softmax(logits_base, dim=-1)

            for dname, dvec in directions.items():
                for alpha in ALPHAS:
                    def make_hook(direction, strength):
                        def hook_fn(module, input, output):
                            return output + strength * direction.to(output.device, output.dtype)
                        return hook_fn

                    block = model.transformer.blocks[LAYER]
                    handle = block.register_forward_hook(make_hook(dvec, alpha))
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out_s = model(sequence_tokens=seq_tokens)
                        logits_s = out_s.sequence_logits.float().cpu()[0, 1:-1]
                    handle.remove()

                    probs_s = torch.softmax(logits_s, dim=-1)
                    kl = torch.sum(probs_base * torch.log(probs_base / (probs_s + 1e-10) + 1e-10), dim=-1).mean().item()
                    results[dname][f"alpha_{alpha}"].append(kl)

        except Exception as e:
            if pi < 3: log.warning(f"Error {acc}: {e}")
            continue

        if (pi + 1) % 50 == 0:
            log.info(f"  {pi+1}/{len(selected)} | {time.time()-t0:.0f}s")

    # Aggregate
    output = {"n_proteins": len(selected), "steering_layer": LAYER, "per_direction": {}}
    for dname in directions:
        output["per_direction"][dname] = {}
        for akey, vals in results[dname].items():
            output["per_direction"][dname][akey] = {
                "mean_kl": float(np.mean(vals)) if vals else 0,
                "std_kl": float(np.std(vals)) if vals else 0,
                "n": len(vals),
            }
        log.info(f"  {dname}: alpha=50 KL={output['per_direction'][dname].get('alpha_50.0',{}).get('mean_kl',0):.6f}")

    out_path = OUTPUT_DIR / "steering_controls.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
