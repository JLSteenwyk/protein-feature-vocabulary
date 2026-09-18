#!/usr/bin/env python3
"""Functional impact of ablating L0H7 (geometric attention head).

Measures what specific capabilities are lost when L0H7 is zeroed:
1. Secondary structure prediction accuracy (SS3: helix/sheet/coil)
2. Structure-enhanced SAE feature activations
3. Contact prediction from attention (if attention capture works post-ablation)

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/scaled_1.5M/34_l0h7_functional_impact.py
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
log = logging.getLogger("l0h7")

EVAL_DIR = ROOT / "data" / "eval_expanded"
STRUCT_DIR = EVAL_DIR / "structures"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
MODEL_ROOT = ROOT / "models" / "sae_1.5M"
N_PROTEINS = 100
SEED = 42

def main():
    from models.esm3_hooks import load_esm3, ESM3HookManager
    from models.interventions import head_ablation
    from esm.sdk.api import ESMProtein
    from sae.model import build_sae

    # Load structures
    with open(STRUCT_DIR / "manifest.json") as f:
        available = json.load(f)["available"]
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    rng = np.random.RandomState(SEED)
    candidates = [acc for acc in sorted(available.keys())
                  if acc in sequences and 50 <= len(sequences[acc]) <= 300]
    selected = rng.choice(candidates, min(N_PROTEINS, len(candidates)), replace=False).tolist()
    log.info(f"Selected {len(selected)} proteins")

    # Load model
    log.info("Loading ESM-3...")
    model, tok = load_esm3(device="cuda")
    log.info("ESM-3 loaded")

    # Load SAE
    ckpt = torch.load(MODEL_ROOT / "esm3_residue_ef8_k64" / "best.pt", map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae = sae.eval().cuda()

    # Enhanced feature IDs
    cm = json.load(open(OUTPUT_DIR / "cross_modal_features.json"))
    enhanced_ids = cm["enhanced_feature_ids"][:50]

    # Results storage
    results = {
        "ss_prediction": {"normal": [], "ablated": []},
        "sae_features": {"normal_enhanced_mean": [], "ablated_enhanced_mean": [],
                         "normal_all_mean": [], "ablated_all_mean": []},
        "token_agreement": [],
    }

    t0 = time.time()
    n_done = 0

    for acc in selected:
        pdb_path = available[acc]
        try:
            prot_sst = ESMProtein.from_pdb(pdb_path)
            if prot_sst.sequence is None: continue
            seq = prot_sst.sequence
            if len(seq) < 50 or len(seq) > 300: continue

            prot_s = ESMProtein(sequence=seq)
            tok_sst = model.encode(prot_sst)
            seq_tokens = tok_sst.sequence.to("cuda").unsqueeze(0)
            struct_tokens = tok_sst.structure.to("cuda").unsqueeze(0)

            # === Normal S+St forward ===
            mgr = ESM3HookManager(model, layers=[33])
            mgr.register()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out_normal = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
            hidden_normal = mgr.cache.residual_stream[33].float().cpu()
            mgr.remove()

            logits_normal = out_normal.sequence_logits.float().cpu()[0, 1:-1]
            ss_logits_normal = out_normal.secondary_structure_logits.float().cpu()[0, 1:-1] if hasattr(out_normal, 'secondary_structure_logits') else None

            # SAE encode normal
            with torch.no_grad():
                z_normal = sae.encode(hidden_normal[0, 1:-1].cuda()).cpu()

            # === Ablated S+St forward (zero L0H7) ===
            mgr2 = ESM3HookManager(model, layers=[33])
            mgr2.register()
            with head_ablation(model, "esm3", 0, 7):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_ablated = model(sequence_tokens=seq_tokens, structure_tokens=struct_tokens)
            hidden_ablated = mgr2.cache.residual_stream[33].float().cpu()
            mgr2.remove()

            logits_ablated = out_ablated.sequence_logits.float().cpu()[0, 1:-1]
            ss_logits_ablated = out_ablated.secondary_structure_logits.float().cpu()[0, 1:-1] if hasattr(out_ablated, 'secondary_structure_logits') else None

            # SAE encode ablated
            with torch.no_grad():
                z_ablated = sae.encode(hidden_ablated[0, 1:-1].cuda()).cpu()

            # === Measure 1: SS prediction ===
            if ss_logits_normal is not None and ss_logits_ablated is not None:
                # SS3 accuracy: compare argmax predictions
                ss_pred_normal = ss_logits_normal[:, :3].argmax(dim=-1)  # H, E, C
                ss_pred_ablated = ss_logits_ablated[:, :3].argmax(dim=-1)
                agreement = (ss_pred_normal == ss_pred_ablated).float().mean().item()
                results["ss_prediction"]["normal"].append(1.0)  # placeholder
                results["ss_prediction"]["ablated"].append(agreement)

            # === Measure 2: SAE feature activations ===
            # Enhanced features
            enh_act_normal = z_normal[:, enhanced_ids].mean().item()
            enh_act_ablated = z_ablated[:, enhanced_ids].mean().item()
            results["sae_features"]["normal_enhanced_mean"].append(enh_act_normal)
            results["sae_features"]["ablated_enhanced_mean"].append(enh_act_ablated)

            # All features
            all_act_normal = z_normal.mean().item()
            all_act_ablated = z_ablated.mean().item()
            results["sae_features"]["normal_all_mean"].append(all_act_normal)
            results["sae_features"]["ablated_all_mean"].append(all_act_ablated)

            # === Measure 3: Token prediction agreement ===
            top_normal = logits_normal.argmax(dim=-1)
            top_ablated = logits_ablated.argmax(dim=-1)
            min_L = min(len(top_normal), len(top_ablated))
            token_agree = (top_normal[:min_L] == top_ablated[:min_L]).float().mean().item()
            results["token_agreement"].append(token_agree)

            n_done += 1
            if n_done % 20 == 0:
                log.info(f"  {n_done}/{len(selected)} | {time.time()-t0:.0f}s")

        except Exception as e:
            if n_done < 3: log.warning(f"Error {acc}: {e}")
            continue
        torch.cuda.empty_cache()

    log.info(f"\nProcessed {n_done} proteins in {(time.time()-t0)/60:.1f} min")

    # Aggregate
    output = {"n_proteins": n_done, "ablated_head": "L0H7"}

    # SS prediction
    if results["ss_prediction"]["ablated"]:
        ss_agree = np.mean(results["ss_prediction"]["ablated"])
        output["ss_prediction"] = {
            "mean_ss3_agreement_after_ablation": float(ss_agree),
            "interpretation": f"After ablating L0H7, {ss_agree*100:.1f}% of SS3 predictions remain unchanged",
        }
        log.info(f"SS3 agreement after L0H7 ablation: {ss_agree*100:.1f}%")

    # SAE features
    enh_normal = np.mean(results["sae_features"]["normal_enhanced_mean"])
    enh_ablated = np.mean(results["sae_features"]["ablated_enhanced_mean"])
    all_normal = np.mean(results["sae_features"]["normal_all_mean"])
    all_ablated = np.mean(results["sae_features"]["ablated_all_mean"])
    output["sae_features"] = {
        "enhanced_mean_normal": float(enh_normal),
        "enhanced_mean_ablated": float(enh_ablated),
        "enhanced_change_frac": float((enh_ablated - enh_normal) / (enh_normal + 1e-10)),
        "all_mean_normal": float(all_normal),
        "all_mean_ablated": float(all_ablated),
        "all_change_frac": float((all_ablated - all_normal) / (all_normal + 1e-10)),
    }
    log.info(f"Enhanced SAE features: normal={enh_normal:.4f}, ablated={enh_ablated:.4f} "
             f"({(enh_ablated-enh_normal)/(enh_normal+1e-10)*100:.1f}% change)")
    log.info(f"All SAE features: normal={all_normal:.4f}, ablated={all_ablated:.4f} "
             f"({(all_ablated-all_normal)/(all_normal+1e-10)*100:.1f}% change)")

    # Token agreement
    token_agree = np.mean(results["token_agreement"])
    output["token_prediction"] = {
        "mean_top1_agreement": float(token_agree),
        "interpretation": f"{token_agree*100:.1f}% of top-1 token predictions unchanged after L0H7 ablation",
    }
    log.info(f"Token prediction agreement: {token_agree*100:.1f}%")

    out_path = OUTPUT_DIR / "l0h7_functional_impact.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
