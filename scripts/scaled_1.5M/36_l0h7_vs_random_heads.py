#!/usr/bin/env python3
"""Compare L0H7 ablation vs 10 randomly selected L0 heads on functional measures.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/36_l0h7_vs_random_heads.py
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
log = logging.getLogger("l0h7_rand")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
N_PROTEINS = 100
SEED = 42

def run_ablation(model, sae, enhanced_ids, proteins, layer, head):
    from models.esm3_hooks import ESM3HookManager
    from models.interventions import head_ablation
    from esm.sdk.api import ESMProtein

    ss_ag, tok_ag, enh_n, enh_a = [], [], [], []
    for acc, pdb_path in proteins:
        try:
            prot = ESMProtein.from_pdb(pdb_path)
            if prot.sequence is None or len(prot.sequence) < 50 or len(prot.sequence) > 300: continue
            tok = model.encode(prot)
            seq_t = tok.sequence.to("cuda").unsqueeze(0)
            str_t = tok.structure.to("cuda").unsqueeze(0)

            mgr = ESM3HookManager(model, layers=[33]); mgr.register()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_n = model(sequence_tokens=seq_t, structure_tokens=str_t)
            h_n = mgr.cache.residual_stream[33].float().cpu(); mgr.remove()

            mgr2 = ESM3HookManager(model, layers=[33]); mgr2.register()
            with head_ablation(model, "esm3", layer, head):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_a = model(sequence_tokens=seq_t, structure_tokens=str_t)
            h_a = mgr2.cache.residual_stream[33].float().cpu(); mgr2.remove()

            l_n = out_n.sequence_logits.float().cpu()[0, 1:-1]
            l_a = out_a.sequence_logits.float().cpu()[0, 1:-1]

            if hasattr(out_n, 'secondary_structure_logits') and out_n.secondary_structure_logits is not None:
                ss_n = out_n.secondary_structure_logits.float().cpu()[0, 1:-1, :3].argmax(-1)
                ss_a = out_a.secondary_structure_logits.float().cpu()[0, 1:-1, :3].argmax(-1)
                ss_ag.append((ss_n == ss_a).float().mean().item())

            mL = min(l_n.size(0), l_a.size(0))
            tok_ag.append((l_n[:mL].argmax(-1) == l_a[:mL].argmax(-1)).float().mean().item())

            with torch.no_grad():
                z_n = sae.encode(h_n[0, 1:-1].cuda()).cpu()
                z_a = sae.encode(h_a[0, 1:-1].cuda()).cpu()
            enh_n.append(z_n[:, enhanced_ids].mean().item())
            enh_a.append(z_a[:, enhanced_ids].mean().item())
        except: continue
        torch.cuda.empty_cache()

    return {
        "ss3_agreement": float(np.mean(ss_ag)) if ss_ag else None,
        "token_agreement": float(np.mean(tok_ag)),
        "enhanced_sae_normal": float(np.mean(enh_n)),
        "enhanced_sae_ablated": float(np.mean(enh_a)),
        "n_proteins": len(tok_ag),
    }

def main():
    from models.esm3_hooks import load_esm3
    from sae.model import build_sae

    with open(STRUCT_DIR / "manifest.json") as f:
        available = json.load(f)["available"]
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [(acc, available[acc]) for acc in sorted(available.keys())
                  if acc in sequences and 50 <= len(sequences[acc]) <= 300]
    selected = [candidates[i] for i in rng.choice(len(candidates), min(N_PROTEINS, len(candidates)), replace=False)]

    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")

    ckpt = torch.load(MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt", map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.eval().cuda()

    cm = json.load(open(OUTPUT_DIR / "cross_modal_features.json"))
    enhanced_ids = cm["enhanced_feature_ids"][:50]

    # Select 10 random L0 heads (excluding H7)
    all_other = [h for h in range(24) if h != 7]
    random_heads = rng.choice(all_other, 10, replace=False).tolist()
    log.info(f"Random L0 heads: {random_heads}")

    # Run L0H7
    log.info("\n=== L0H7 ===")
    r_h7 = run_ablation(model, sae, enhanced_ids, selected, 0, 7)
    log.info(f"  SS3={r_h7['ss3_agreement']:.3f}, Token={r_h7['token_agreement']:.3f}, "
             f"Enh SAE: {r_h7['enhanced_sae_ablated']:.3f}/{r_h7['enhanced_sae_normal']:.3f}")

    # Run each random head
    random_results = []
    for h in random_heads:
        log.info(f"\n=== L0H{h} ===")
        r = run_ablation(model, sae, enhanced_ids, selected, 0, h)
        log.info(f"  SS3={r['ss3_agreement']:.3f}, Token={r['token_agreement']:.3f}, "
                 f"Enh SAE: {r['enhanced_sae_ablated']:.3f}/{r['enhanced_sae_normal']:.3f}")
        r["head"] = h
        random_results.append(r)

    # Aggregate random heads
    rand_ss3 = [r["ss3_agreement"] for r in random_results if r["ss3_agreement"] is not None]
    rand_tok = [r["token_agreement"] for r in random_results]
    rand_enh_frac = [r["enhanced_sae_ablated"] / (r["enhanced_sae_normal"] + 1e-10) for r in random_results]

    log.info(f"\n=== SUMMARY ===")
    log.info(f"L0H7:   SS3={r_h7['ss3_agreement']:.3f}, Token={r_h7['token_agreement']:.3f}")
    log.info(f"Random: SS3={np.mean(rand_ss3):.3f}±{np.std(rand_ss3):.3f}, "
             f"Token={np.mean(rand_tok):.3f}±{np.std(rand_tok):.3f}")

    output = {
        "l0h7": r_h7,
        "random_heads": random_heads,
        "random_results": random_results,
        "random_summary": {
            "mean_ss3_agreement": float(np.mean(rand_ss3)),
            "std_ss3_agreement": float(np.std(rand_ss3)),
            "mean_token_agreement": float(np.mean(rand_tok)),
            "std_token_agreement": float(np.std(rand_tok)),
            "mean_enh_sae_frac": float(np.mean(rand_enh_frac)),
            "std_enh_sae_frac": float(np.std(rand_enh_frac)),
        },
    }

    out_path = OUTPUT_DIR / "l0h7_vs_random_heads.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
