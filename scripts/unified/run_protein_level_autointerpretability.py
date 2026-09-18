#!/usr/bin/env python3
"""Protein-level autointerpretability: replicating InterPLM/Gujral protocol.

Computes protein-level SAE feature activations (mean across residues),
generates LLM descriptions, validates by predicting activation on held-out
proteins. Compares cross-modal vs matched interpretability scores.

Protocol (following Gujral et al. PNAS 2025):
1. For each feature: compute mean activation per protein
2. Select top-17 active + 15 inactive proteins → generate description via LLM
3. Validation: give LLM description + held-out protein metadata → predict activation
4. Compute Pearson r between predicted and actual → autointerpretability score

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/run_protein_level_autointerpretability.py
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

from sae.model import TopKSAE, build_sae

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("prot_interp")


def load_sae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def query_claude(prompt, max_tokens=1500):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(model="claude-sonnet-4-6", max_tokens=max_tokens,
                                   messages=[{"role": "user", "content": prompt}],
                                   system="You are a helpful assistant. Always respond with valid JSON only, no additional text.")
    return resp.content[0].text


def query_openai(prompt, max_tokens=1500):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(model="gpt-5.4", max_completion_tokens=max_tokens,
                                           messages=[{"role": "user", "content": prompt}])
    return resp.choices[0].message.content


def parse_json_response(text):
    import re
    text = text.strip()
    # Strip markdown code blocks
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find JSON object
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    # Try to find JSON list
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    # Last resort: try to extract comma-separated numbers
    nums = re.findall(r"[\d.]+", text)
    if len(nums) >= 5:
        try:
            return [float(n) for n in nums]
        except ValueError:
            pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-features", type=int, default=0,
                        help="Features to analyze (0=all active)")
    parser.add_argument("--n-describe", type=int, default=3500,
                        help="Proteins for generating descriptions")
    parser.add_argument("--n-example-active", type=int, default=17,
                        help="Active proteins shown in description prompt")
    parser.add_argument("--n-example-inactive", type=int, default=15,
                        help="Inactive proteins shown in description prompt")
    parser.add_argument("--n-validate", type=int, default=30,
                        help="Held-out proteins per feature for validation")
    args = parser.parse_args()

    rng = np.random.RandomState(42)

    # Load protein-level SAE
    sae_path = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_protein_level_topk" / "best.pt"
    ckpt = torch.load(str(sae_path), map_location="cpu", weights_only=False)
    sae = build_sae(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    log.info(f"Loaded protein-level SAE: {ckpt['sae_config']}")

    # Load protein-level activations
    h5_path = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33_protein_level.h5"
    with h5py.File(str(h5_path), "r") as f:
        X = torch.tensor(f["activations"][:], dtype=torch.float32)
        all_accs = [a.decode() for a in f["accessions"][:]]
    log.info(f"Protein-level activations: {X.shape}, {len(all_accs)} proteins")

    # Encode through protein-level SAE
    with torch.no_grad():
        Z = sae.encode(X).numpy()  # (N_proteins, d_sae)
    log.info(f"SAE features: {Z.shape}")

    # Find active features (non-dead)
    active_mask = (Z > 0).any(axis=0)
    active_features = np.where(active_mask)[0].tolist()
    log.info(f"Active features: {len(active_features)} / {Z.shape[1]}")

    if args.n_features > 0:
        all_features = active_features[:args.n_features]
    else:
        all_features = active_features
    # No cross-modal/matched split for protein-level SAE — label all as "protein_level"
    cm_set = set()  # empty — all features are protein-level
    log.info(f"Analyzing {len(all_features)} features")

    # Load metadata
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    # Split describe/validate
    describe_accs = all_accs[:args.n_describe]
    validate_accs = all_accs[args.n_describe:]
    log.info(f"Description proteins: {len(describe_accs)}, Validation proteins: {len(validate_accs)}")

    # Build acc→index mapping
    acc_to_idx = {acc: i for i, acc in enumerate(all_accs)}

    # ═══════════════════════════════════════════════════
    # Build protein-level feature activations
    # ═══════════════════════════════════════════════════
    log.info("\nBuilding protein-level feature activations...")

    # {fid: {acc: activation}}
    feature_prot_acts = {fid: {} for fid in all_features}
    for fid in all_features:
        for acc in all_accs:
            idx = acc_to_idx[acc]
            feature_prot_acts[fid][acc] = float(Z[idx, fid])

    log.info("  Done.")

    # ═══════════════════════════════════════════════════
    # Generate descriptions and validate per feature
    # ═══════════════════════════════════════════════════
    log.info("\nGenerating descriptions and validating...")

    results = []

    for fi, fid in enumerate(all_features):
        label = "protein_level"

        # Description set: rank by activation among describe_accs
        desc_acts = [(acc, feature_prot_acts[fid].get(acc, 0)) for acc in describe_accs]
        desc_acts.sort(key=lambda x: -x[1])

        # Global max for normalization
        all_vals = [v for v in feature_prot_acts[fid].values()]
        global_max = max(all_vals) if all_vals and max(all_vals) > 0 else 1.0

        # Top active proteins
        top_active = desc_acts[:args.n_example_active]
        # Bottom inactive proteins (low activation)
        bottom_inactive = desc_acts[-args.n_example_inactive:]

        # Build description prompt (protein-level, following Gujral protocol)
        active_list = ""
        for i, (acc, act) in enumerate(top_active):
            meta = metadata.get(acc, {})
            name = meta.get("protein_name", acc)
            org = meta.get("organism", "")
            pfam = meta.get("pfam", [])
            pfam_str = ", ".join(pfam[:3]) if pfam else "N/A"
            func = ", ".join(meta.get("function", [])[:2]) if meta.get("function") else "N/A"
            go_terms = [g.get("term", "").split(":")[-1].strip() for g in meta.get("go_terms", [])[:4]]
            go_str = "; ".join(go_terms) if go_terms else "N/A"
            norm_act = act / global_max
            active_list += (
                f"  {i+1}. {name} ({acc}, {org})\n"
                f"     Activation: {norm_act:.3f} | Pfam: {pfam_str}\n"
                f"     Function: {func}\n"
                f"     GO terms: {go_str}\n"
            )

        inactive_list = ""
        for i, (acc, act) in enumerate(bottom_inactive):
            meta = metadata.get(acc, {})
            name = meta.get("protein_name", acc)
            org = meta.get("organism", "")
            pfam = meta.get("pfam", [])
            pfam_str = ", ".join(pfam[:3]) if pfam else "N/A"
            func = ", ".join(meta.get("function", [])[:2]) if meta.get("function") else "N/A"
            norm_act = act / global_max
            inactive_list += (
                f"  {i+1}. {name} ({acc}, {org})\n"
                f"     Activation: {norm_act:.3f} | Pfam: {pfam_str} | Function: {func}\n"
            )

        desc_prompt = f"""You are an expert protein biochemist. A sparse autoencoder feature extracted from a protein language model activates on certain proteins more than others. Based on the proteins below, provide a concise interpretation of what this feature detects.

## Highly activating proteins (ranked by activation strength):
{active_list}

## Low/non-activating proteins:
{inactive_list}

What biological pattern, protein family, function, or property does this feature detect? Provide a single, specific interpretation in 1-2 sentences. Focus on what the highly activating proteins have in common that the inactive ones lack.

Respond in JSON: {{"description": "...", "category": "protein_family|function|localization|structure|metabolism|unknown", "confidence": "high|medium|low"}}"""

        # Generate description (Claude only for speed)
        try:
            desc_resp = query_claude(desc_prompt, max_tokens=300)
            desc_parsed = parse_json_response(desc_resp)
            if desc_parsed is None:
                description = desc_resp[:500]
                category = "unknown"
            else:
                description = desc_parsed.get("description", desc_resp[:500])
                category = desc_parsed.get("category", "unknown")
        except Exception as e:
            log.warning(f"  Description error f/{fid}: {e}")
            description = f"Error: {e}"
            category = "unknown"

        # ═══════════════════════════════════════════════
        # Validate: predict activation on held-out proteins
        # ═══════════════════════════════════════════════
        val_acts = [(acc, feature_prot_acts[fid].get(acc, 0)) for acc in validate_accs
                     if acc in feature_prot_acts[fid]]

        if len(val_acts) < args.n_validate:
            val_selected = val_acts
        else:
            # Select diverse range
            val_acts.sort(key=lambda x: -x[1])
            n = len(val_acts)
            top = val_acts[:8]
            mid = val_acts[n//3:n//3 + 7]
            bottom = val_acts[-8:]
            remaining = list(set(range(n)) - set(range(8)) - set(range(n//3, n//3+7)) - set(range(n-8, n)))
            if remaining:
                extra_idx = rng.choice(remaining, min(7, len(remaining)), replace=False)
                extra = [val_acts[i] for i in extra_idx]
            else:
                extra = []
            val_selected = top + mid + bottom + extra
            seen = set()
            deduped = []
            for v in val_selected:
                if v[0] not in seen:
                    seen.add(v[0])
                    deduped.append(v)
            val_selected = deduped[:args.n_validate]

        actual_values = [v[1] / global_max for v in val_selected]

        # Build validation prompt
        val_protein_list = ""
        for j, (acc, act) in enumerate(val_selected):
            meta = metadata.get(acc, {})
            name = meta.get("protein_name", acc)
            org = meta.get("organism", "")
            pfam = meta.get("pfam", [])
            pfam_str = ", ".join(pfam[:3]) if pfam else "N/A"
            func = ", ".join(meta.get("function", [])[:2]) if meta.get("function") else "N/A"
            go_terms = [g.get("term", "").split(":")[-1].strip() for g in meta.get("go_terms", [])[:3]]
            go_str = "; ".join(go_terms) if go_terms else "N/A"
            val_protein_list += (
                f"  {j+1}. {name} ({acc}, {org})\n"
                f"     Pfam: {pfam_str} | Function: {func} | GO: {go_str}\n"
            )

        val_prompt = f"""A protein language model feature has been described as:
"{description}"

For each protein below, predict how strongly this feature activates (0.0 = no activation, 1.0 = maximum). Base your prediction on how well the protein matches the feature description.

## Proteins:
{val_protein_list}

Return ONLY a JSON list of {len(val_selected)} numbers (predicted activations in order):
[pred_1, pred_2, ..., pred_{len(val_selected)}]"""

        # Validate with both LLMs
        feature_result = {
            "feature_id": fid,
            "label": label,
            "description": description,
            "category": category,
            "n_validate": len(val_selected),
            "actual_values": actual_values,
        }

        for model_name, query_fn in [("claude", query_claude), ("gpt", query_openai)]:
            try:
                resp = query_fn(val_prompt, max_tokens=500)
                parsed = parse_json_response(resp)
                if isinstance(parsed, list) and len(parsed) >= 3:
                    # Use as many predictions as we got (pad with 0.5 if needed)
                    preds = []
                    for p in parsed[:len(val_selected)]:
                        try:
                            preds.append(float(p))
                        except (ValueError, TypeError):
                            preds.append(0.5)
                    while len(preds) < len(val_selected):
                        preds.append(0.5)
                    preds = preds[:len(val_selected)]
                    r, p = stats.pearsonr(actual_values, preds)
                    feature_result[f"{model_name}_predictions"] = preds
                    feature_result[f"{model_name}_pearson_r"] = float(r)
                    feature_result[f"{model_name}_pearson_p"] = float(p)
                else:
                    feature_result[f"{model_name}_pearson_r"] = None
            except Exception as e:
                log.warning(f"  {model_name} validation error f/{fid}: {e}")
                feature_result[f"{model_name}_pearson_r"] = None

        results.append(feature_result)

        if (fi + 1) % 10 == 0:
            valid_claude = [r["claude_pearson_r"] for r in results if r.get("claude_pearson_r") is not None]
            valid_gpt = [r["gpt_pearson_r"] for r in results if r.get("gpt_pearson_r") is not None]
            cm_claude = [r["claude_pearson_r"] for r in results
                         if r.get("claude_pearson_r") is not None and r["label"] == "cross-modal"]
            mt_claude = [r["claude_pearson_r"] for r in results
                         if r.get("claude_pearson_r") is not None and r["label"] == "matched"]

            log.info(f"  Feature {fi+1}/{len(all_features)} | "
                     f"Claude median r={np.median(valid_claude):.3f} (n={len(valid_claude)}) | "
                     f"GPT median r={np.median(valid_gpt):.3f} (n={len(valid_gpt)})"
                     + (f" | CM r={np.median(cm_claude):.3f}, MT r={np.median(mt_claude):.3f}"
                        if cm_claude and mt_claude else ""))
            time.sleep(0.3)

    # ═══════════════════════════════════════════════════
    # Summary statistics
    # ═══════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("PROTEIN-LEVEL AUTOINTERPRETABILITY RESULTS")
    log.info("=" * 60)

    for model in ["claude", "gpt"]:
        rkey = f"{model}_pearson_r"
        valid = [r[rkey] for r in results if r.get(rkey) is not None and not np.isnan(r[rkey])]
        cm_r = [r[rkey] for r in results if r.get(rkey) is not None and not np.isnan(r[rkey])
                and r["label"] == "cross-modal"]
        mt_r = [r[rkey] for r in results if r.get(rkey) is not None and not np.isnan(r[rkey])
                and r["label"] == "matched"]

        log.info(f"\n--- {model.upper()} ---")
        log.info(f"Overall: median r = {np.median(valid):.3f}, mean r = {np.mean(valid):.3f} "
                 f"(n={len(valid)})")
        if cm_r:
            log.info(f"Cross-modal: median r = {np.median(cm_r):.3f}, mean r = {np.mean(cm_r):.3f} "
                     f"(n={len(cm_r)})")
        if mt_r:
            log.info(f"Matched:     median r = {np.median(mt_r):.3f}, mean r = {np.mean(mt_r):.3f} "
                     f"(n={len(mt_r)})")

        if cm_r and mt_r:
            stat, pval = stats.mannwhitneyu(cm_r, mt_r, alternative="two-sided")
            log.info(f"CM vs MT: Mann-Whitney p = {pval:.4f}")

        # Distribution of r values
        log.info(f"r distribution:")
        for lo, hi in [(-1, 0), (0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
            n = sum(1 for r in valid if lo <= r < hi)
            log.info(f"  [{lo:.2f}, {hi:.2f}): {n} ({100*n/max(len(valid),1):.0f}%)")

    # Category distribution
    all_cats = Counter(r["category"] for r in results)
    log.info(f"\nCategory distribution: {dict(all_cats)}")

    # Sample descriptions
    log.info(f"\nSample descriptions:")
    for r in results[:10]:
        log.info(f"  f/{r['feature_id']}: {r['description'][:150]}")

    # ═══════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════
    summary = {}
    for model in ["claude", "gpt"]:
        rkey = f"{model}_pearson_r"
        valid = [r[rkey] for r in results if r.get(rkey) is not None and not np.isnan(r[rkey])]
        cm_r = [r[rkey] for r in results if r.get(rkey) is not None and not np.isnan(r[rkey])
                and r["label"] == "cross-modal"]
        mt_r = [r[rkey] for r in results if r.get(rkey) is not None and not np.isnan(r[rkey])
                and r["label"] == "matched"]
        mw_p = float(stats.mannwhitneyu(cm_r, mt_r, alternative="two-sided")[1]) if cm_r and mt_r else None
        summary[model] = {
            "overall_median_r": float(np.median(valid)) if valid else None,
            "overall_mean_r": float(np.mean(valid)) if valid else None,
            "crossmodal_median_r": float(np.median(cm_r)) if cm_r else None,
            "matched_median_r": float(np.median(mt_r)) if mt_r else None,
            "cm_vs_mt_p": mw_p,
            "n_valid": len(valid),
        }

    output = {
        "method": "Protein-level autointerpretability (Gujral/InterPLM protocol)",
        "n_features": len(results),
        "n_describe_proteins": len(describe_accs),
        "n_validate_proteins_per_feature": args.n_validate,
        "summary": summary,
        "per_feature_results": results,
    }

    def make_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_safe(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    out_path = ROOT / "results" / "unified" / "crossmodal_features" / "protein_level_autointerpretability.json"
    with open(out_path, "w") as f:
        json.dump(make_safe(output), f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
