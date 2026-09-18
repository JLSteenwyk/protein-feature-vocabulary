#!/usr/bin/env python3
"""InterPLM-style validation: can LLM descriptions predict feature activation on held-out proteins?

Following InterPLM (Simon & Zou, Nature Methods 2025, Fig. 4):
1. Use existing LLM-generated feature descriptions
2. For held-out proteins, ask LLM to predict activation level from description + protein metadata
3. Compare predicted vs actual activation → Pearson r per feature
4. Compare prediction quality between cross-modal and matched features

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/run_llm_activation_prediction.py
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
from scipy import stats

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llm_predict")


def load_sae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def query_claude(prompt, max_tokens=2048):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(model="claude-opus-4-6", max_tokens=max_tokens,
                                   messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text


def query_openai(prompt, max_tokens=2048):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(model="gpt-5.4", max_completion_tokens=max_tokens,
                                           messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def parse_predictions(text, n_expected):
    """Parse predicted activation values from LLM response."""
    text = text.strip()
    # Try to find JSON array or object
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    
    # Try JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "predictions" in parsed:
            return parsed["predictions"]
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in text
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    
    # Fallback: try to extract numbers line by line
    predictions = []
    for line in text.split("\n"):
        # Look for patterns like "1. 0.7" or "protein_id: 0.5"
        import re
        nums = re.findall(r':\s*([\d.]+)', line)
        if nums:
            try:
                predictions.append(float(nums[-1]))
            except ValueError:
                pass
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-test-proteins", type=int, default=20,
                        help="Number of held-out proteins per feature")
    parser.add_argument("--n-features", type=int, default=200,
                        help="Total features to validate (from existing annotations)")
    args = parser.parse_args()

    # Load existing annotations
    ann_path = ROOT / "results" / "unified" / "crossmodal_features" / "llm_feature_annotations.json"
    with open(ann_path) as f:
        ann_data = json.load(f)
    
    annotations_claude = ann_data["annotations_claude"]
    annotations_gpt = ann_data["annotations_gpt"]
    
    log.info(f"Loaded {len(annotations_claude)} existing annotations")

    # Build feature_id → description lookup
    claude_descs = {}
    gpt_descs = {}
    feature_labels = {}
    for ac, ag in zip(annotations_claude, annotations_gpt):
        fid = ac["feature_id"]
        claude_descs[fid] = ac["annotation"].get("description", "")
        gpt_descs[fid] = ag["annotation"].get("description", "")
        feature_labels[fid] = ac["label"]

    # Load SAE and data
    sae_path = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_scaled_topk" / "best.pt"
    sae = load_sae(str(sae_path))
    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33.h5")

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    all_accs = sorted(sequences.keys())
    offsets = {}
    pos = 0
    for acc in all_accs:
        L = len(sequences[acc])
        offsets[acc] = (pos, L)
        pos += L

    # Split: first 400 were used for description generation, use 401-600+ as held-out
    train_accs = set(all_accs[:400])
    test_accs = all_accs[400:700]  # Use next 300 as test pool
    log.info(f"Train proteins: {len(train_accs)}, Test pool: {len(test_accs)}")

    # ═══════════════════════════════════════════════════
    # Compute actual activation values for all test proteins
    # ═══════════════════════════════════════════════════
    log.info("\nComputing actual activations for test proteins...")

    feature_ids = list(claude_descs.keys())[:args.n_features]
    
    # Per-feature: {fid: [(acc, max_activation, frac_active), ...]}
    test_activations = {fid: [] for fid in feature_ids}

    with h5py.File(h5_path, "r") as f:
        acts_data = f["activations"]
        total_residues = acts_data.shape[0]

        for pi, acc in enumerate(test_accs):
            if acc not in offsets:
                continue
            start, L = offsets[acc]
            if start + L > total_residues or L < 5:
                continue

            raw = torch.tensor(acts_data[start:start + L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()

            for fid in feature_ids:
                col = z[:, fid]
                max_act = float(col.max())
                frac_act = float((col > 0).sum()) / L
                test_activations[fid].append({
                    "acc": acc,
                    "max_act": max_act,
                    "frac_active": frac_act,
                })

            if (pi + 1) % 100 == 0:
                log.info(f"  Scanned {pi + 1}/{len(test_accs)} test proteins")

    log.info(f"  Test activation scan complete")

    # ═══════════════════════════════════════════════════
    # For each feature, select held-out proteins with range of activations
    # ═══════════════════════════════════════════════════
    n_test = args.n_test_proteins
    rng = np.random.RandomState(123)

    feature_test_sets = {}
    for fid in feature_ids:
        acts = test_activations[fid]
        if len(acts) < n_test:
            continue

        # Sort by max_act
        acts.sort(key=lambda x: -x["max_act"])

        # Get global max for normalization
        global_max = acts[0]["max_act"] if acts[0]["max_act"] > 0 else 1.0

        # Select diverse range: top 5, middle 5, bottom 5, random 5
        top = acts[:5]
        n = len(acts)
        mid_start = n // 3
        mid = acts[mid_start:mid_start + 5]
        bottom = acts[-5:]
        
        # Random from remaining
        remaining_idx = list(set(range(n)) - set(range(5)) - set(range(mid_start, mid_start+5)) - set(range(n-5, n)))
        if len(remaining_idx) >= 5:
            rand_idx = rng.choice(remaining_idx, 5, replace=False)
            rand_picks = [acts[i] for i in rand_idx]
        else:
            rand_picks = []

        selected = top + mid + bottom + rand_picks
        # Deduplicate by accession
        seen = set()
        deduped = []
        for s in selected:
            if s["acc"] not in seen:
                seen.add(s["acc"])
                deduped.append(s)
        selected = deduped[:n_test]

        # Normalize activations to 0-1
        for s in selected:
            s["normalized_act"] = s["max_act"] / global_max

        feature_test_sets[fid] = {
            "proteins": selected,
            "global_max": global_max,
        }

    log.info(f"Prepared test sets for {len(feature_test_sets)} features")

    # ═══════════════════════════════════════════════════
    # Query LLMs: predict activation for held-out proteins
    # ═══════════════════════════════════════════════════
    log.info("\nQuerying LLMs for activation predictions...")

    results = []
    
    for i, fid in enumerate(feature_test_sets.keys()):
        test_set = feature_test_sets[fid]
        proteins = test_set["proteins"]
        label = feature_labels[fid]
        
        # Build protein list for the prompt
        protein_info = ""
        actual_values = []
        protein_accs = []
        for j, prot in enumerate(proteins):
            acc = prot["acc"]
            meta = metadata.get(acc, {})
            name = meta.get("protein_name", acc)
            org = meta.get("organism", "")
            func_desc = ", ".join(meta.get("function", [])[:2]) if meta.get("function") else "N/A"
            go_terms = [g.get("term", "") for g in meta.get("go_terms", [])[:3]]
            go_str = "; ".join(go_terms[:3]) if go_terms else "N/A"
            length = len(sequences.get(acc, ""))
            
            protein_info += (
                f"  {j+1}. {name} ({acc}, {org})\n"
                f"     Length: {length} residues\n"
                f"     Function: {func_desc}\n"
                f"     GO terms: {go_str}\n\n"
            )
            actual_values.append(prot["normalized_act"])
            protein_accs.append(acc)

        claude_desc = claude_descs[fid]
        
        prompt = f"""You are an expert protein biochemist. A sparse autoencoder feature from a protein language model has been described as follows:

**Feature description**: "{claude_desc}"

For each protein below, predict how strongly this feature would activate (0.0 = no activation, 1.0 = maximum activation). Base your prediction on how well each protein matches the feature description.

## Proteins to score:
{protein_info}

Predict the activation level (0.0 to 1.0) for each protein. Respond as a JSON list of numbers in order:
[prediction_1, prediction_2, ..., prediction_{len(proteins)}]

Return ONLY the JSON list, nothing else."""

        # Query Claude
        try:
            claude_resp = query_claude(prompt, max_tokens=500)
            claude_preds = parse_predictions(claude_resp, len(proteins))
            if isinstance(claude_preds, list) and len(claude_preds) >= len(proteins):
                claude_preds = [float(p) if isinstance(p, (int, float)) else float(p.get("prediction", 0.5) if isinstance(p, dict) else 0.5) for p in claude_preds[:len(proteins)]]
            else:
                claude_preds = None
        except Exception as e:
            log.warning(f"  Claude error for f/{fid}: {e}")
            claude_preds = None

        # Query GPT with same prompt
        try:
            gpt_resp = query_openai(prompt, max_tokens=500)
            gpt_preds = parse_predictions(gpt_resp, len(proteins))
            if isinstance(gpt_preds, list) and len(gpt_preds) >= len(proteins):
                gpt_preds = [float(p) if isinstance(p, (int, float)) else float(p.get("prediction", 0.5) if isinstance(p, dict) else 0.5) for p in gpt_preds[:len(proteins)]]
            else:
                gpt_preds = None
        except Exception as e:
            log.warning(f"  GPT error for f/{fid}: {e}")
            gpt_preds = None

        # Compute correlations
        result = {
            "feature_id": fid,
            "label": label,
            "n_test_proteins": len(proteins),
            "actual_values": actual_values,
        }

        if claude_preds and len(claude_preds) == len(actual_values):
            r, p = stats.pearsonr(actual_values, claude_preds)
            result["claude_predictions"] = claude_preds
            result["claude_pearson_r"] = float(r)
            result["claude_pearson_p"] = float(p)
        else:
            result["claude_predictions"] = None
            result["claude_pearson_r"] = None

        if gpt_preds and len(gpt_preds) == len(actual_values):
            r, p = stats.pearsonr(actual_values, gpt_preds)
            result["gpt_predictions"] = gpt_preds
            result["gpt_pearson_r"] = float(r)
            result["gpt_pearson_p"] = float(p)
        else:
            result["gpt_predictions"] = None
            result["gpt_pearson_r"] = None

        results.append(result)

        if (i + 1) % 10 == 0:
            # Print running stats
            valid_claude = [r["claude_pearson_r"] for r in results if r["claude_pearson_r"] is not None]
            valid_gpt = [r["gpt_pearson_r"] for r in results if r["gpt_pearson_r"] is not None]
            log.info(f"  Validated {i+1}/{len(feature_test_sets)} features | "
                     f"Claude median r={np.median(valid_claude):.3f} (n={len(valid_claude)}) | "
                     f"GPT median r={np.median(valid_gpt):.3f} (n={len(valid_gpt)})")
            time.sleep(0.5)

    # ═══════════════════════════════════════════════════
    # Analyze results
    # ═══════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("ACTIVATION PREDICTION RESULTS")
    log.info("=" * 60)

    for model_name, r_key in [("Claude", "claude_pearson_r"), ("GPT", "gpt_pearson_r")]:
        valid = [(r["feature_id"], r["label"], r[r_key]) for r in results if r[r_key] is not None]
        if not valid:
            log.info(f"\n{model_name}: no valid predictions")
            continue

        all_r = [v[2] for v in valid]
        cm_r = [v[2] for v in valid if v[1] == "cross-modal"]
        mt_r = [v[2] for v in valid if v[1] == "matched"]

        log.info(f"\n--- {model_name} ---")
        log.info(f"Overall: median r = {np.median(all_r):.3f}, mean r = {np.mean(all_r):.3f} "
                 f"(n={len(all_r)})")
        log.info(f"Cross-modal: median r = {np.median(cm_r):.3f}, mean r = {np.mean(cm_r):.3f} "
                 f"(n={len(cm_r)})")
        log.info(f"Matched:     median r = {np.median(mt_r):.3f}, mean r = {np.mean(mt_r):.3f} "
                 f"(n={len(mt_r)})")

        # Test if matched features have better descriptions
        stat, pval = stats.mannwhitneyu(cm_r, mt_r, alternative="two-sided")
        log.info(f"CM vs MT: Mann-Whitney U p = {pval:.4f}")
        
        # Distribution of r values
        bins = [(-1, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        log.info(f"Distribution of r values:")
        for lo, hi in bins:
            n = sum(1 for r in all_r if lo <= r < hi)
            log.info(f"  [{lo:.2f}, {hi:.2f}): {n} features ({100*n/len(all_r):.0f}%)")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    output = {
        "method": "InterPLM-style activation prediction validation",
        "n_features": len(results),
        "n_test_proteins_per_feature": args.n_test_proteins,
        "test_protein_pool": "proteins 401-700 (held out from description generation)",
        "results": results,
        "summary": {},
    }

    for model_name, r_key in [("claude", "claude_pearson_r"), ("gpt", "gpt_pearson_r")]:
        valid = [r[r_key] for r in results if r[r_key] is not None]
        cm_r = [r[r_key] for r in results if r[r_key] is not None and r["label"] == "cross-modal"]
        mt_r = [r[r_key] for r in results if r[r_key] is not None and r["label"] == "matched"]
        
        if valid:
            stat, pval = stats.mannwhitneyu(cm_r, mt_r, alternative="two-sided") if cm_r and mt_r else (0, 1)
            output["summary"][model_name] = {
                "overall_median_r": float(np.median(valid)),
                "overall_mean_r": float(np.mean(valid)),
                "crossmodal_median_r": float(np.median(cm_r)) if cm_r else None,
                "matched_median_r": float(np.median(mt_r)) if mt_r else None,
                "cm_vs_mt_mannwhitney_p": float(pval),
                "n_valid": len(valid),
                "n_crossmodal": len(cm_r),
                "n_matched": len(mt_r),
            }

    out_path = ROOT / "results" / "unified" / "crossmodal_features" / "llm_activation_prediction.json"
    
    def make_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_safe(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    with open(out_path, "w") as f:
        json.dump(make_safe(output), f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
