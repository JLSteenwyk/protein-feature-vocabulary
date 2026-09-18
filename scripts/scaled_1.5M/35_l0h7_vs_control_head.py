#!/usr/bin/env python3
"""Compare L0H7 ablation vs control head (L0H10) ablation on functional measures.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/35_l0h7_vs_control_head.py
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
log = logging.getLogger("l0h7_ctrl")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
N_PROTEINS = 100
SEED = 42
CONTROL_HEAD = 10  # L0H10: KL=0.0002, essentially zero importance

def run_ablation(model, sae, enhanced_ids, proteins, layer, head, device="cuda"):
    """Run ablation of a specific head and measure functional impact."""
    from models.esm3_hooks import ESM3HookManager
    from models.interventions import head_ablation
    from esm.sdk.api import ESMProtein

    ss_agreements = []
    token_agreements = []
    enh_normal_list = []
    enh_ablated_list = []

    for acc, pdb_path in proteins:
        try:
            prot_sst = ESMProtein.from_pdb(pdb_path)
            if prot_sst.sequence is None: continue
            if len(prot_sst.sequence) < 50 or len(prot_sst.sequence) > 300: continue

            tok_sst = model.encode(prot_sst)
            seq_tokens = tok_sst.sequence.to(device).unsqueeze(0)
            struct_tokens = tok_sst.structure.to(device).unsqueeze(0)

            # Normal
            mgr = ESM3HookManager(model, layers=[33])
            mgr.register()
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                out_n = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
            h_n = mgr.cache.residual_stream[33].float().cpu()
            mgr.remove()

            # Ablated
            mgr2 = ESM3HookManager(model, layers=[33])
            mgr2.register()
            with head_ablation(model, "esm3", layer, head):
                with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                    out_a = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
            h_a = mgr2.cache.residual_stream[33].float().cpu()
            mgr2.remove()

            logits_n = out_n.sequence_logits.float().cpu()[0, 1:-1]
            logits_a = out_a.sequence_logits.float().cpu()[0, 1:-1]

            # SS3
            if hasattr(out_n, 'secondary_structure_logits') and out_n.secondary_structure_logits is not None:
                ss_n = out_n.secondary_structure_logits.float().cpu()[0, 1:-1, :3].argmax(dim=-1)
                ss_a = out_a.secondary_structure_logits.float().cpu()[0, 1:-1, :3].argmax(dim=-1)
                ss_agreements.append((ss_n == ss_a).float().mean().item())

            # Token
            min_L = min(logits_n.size(0), logits_a.size(0))
            token_agreements.append((logits_n[:min_L].argmax(-1) == logits_a[:min_L].argmax(-1)).float().mean().item())

            # SAE features
            with torch.no_grad():
                z_n = sae.encode(h_n[0, 1:-1].cuda()).cpu()
                z_a = sae.encode(h_a[0, 1:-1].cuda()).cpu()
            enh_normal_list.append(z_n[:, enhanced_ids].mean().item())
            enh_ablated_list.append(z_a[:, enhanced_ids].mean().item())

        except Exception:
            continue
        torch.cuda.empty_cache()

    return {
        "ss3_agreement": float(np.mean(ss_agreements)) if ss_agreements else None,
        "token_agreement": float(np.mean(token_agreements)),
        "enhanced_sae_normal": float(np.mean(enh_normal_list)),
        "enhanced_sae_ablated": float(np.mean(enh_ablated_list)),
        "n_proteins": len(token_agreements),
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

    log.info(f"Selected {len(selected)} proteins")
    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")

    ckpt = torch.load(MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt", map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.eval().cuda()

    cm = json.load(open(OUTPUT_DIR / "cross_modal_features.json"))
    enhanced_ids = cm["enhanced_feature_ids"][:50]

    log.info(f"\n=== Ablating L0H7 (geometric attention) ===")
    r_h7 = run_ablation(model, sae, enhanced_ids, selected, 0, 7)
    log.info(f"  SS3={r_h7['ss3_agreement']:.3f}, Token={r_h7['token_agreement']:.3f}, "
             f"Enh SAE: {r_h7['enhanced_sae_normal']:.3f} -> {r_h7['enhanced_sae_ablated']:.3f}")

    log.info(f"\n=== Ablating L0H{CONTROL_HEAD} (control) ===")
    r_ctrl = run_ablation(model, sae, enhanced_ids, selected, 0, CONTROL_HEAD)
    log.info(f"  SS3={r_ctrl['ss3_agreement']:.3f}, Token={r_ctrl['token_agreement']:.3f}, "
             f"Enh SAE: {r_ctrl['enhanced_sae_normal']:.3f} -> {r_ctrl['enhanced_sae_ablated']:.3f}")

    output = {
        "l0h7": r_h7,
        "control_l0h10": r_ctrl,
        "control_head": f"L0H{CONTROL_HEAD}",
    }

    out_path = OUTPUT_DIR / "l0h7_vs_control.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
