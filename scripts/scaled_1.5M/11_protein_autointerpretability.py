#!/usr/bin/env python3
"""Protein-level autointerpretability following Gujral/InterPLM protocol.

Uses protein-level SAE features. For each active feature:
1. Show top-active + inactive proteins to LLM → get description
2. Show held-out proteins to LLM → predict activations
3. Compute Pearson r between predicted and actual

Usage:
    ./env/bin/python scripts/scaled_1.5M/11_protein_autointerpretability.py
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
from scipy import stats

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("autointerp")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

N_DESCRIBE = 9000  # proteins for generating descriptions
N_EXAMPLE_ACTIVE = 17
N_EXAMPLE_INACTIVE = 15
N_VALIDATE = 30
MAX_FEATURES = 300  # cap on features to analyze (API cost control)


def query_claude(prompt, max_tokens=300):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        system="You are a helpful assistant. Always respond with valid JSON only, no additional text.",
    )
    return resp.content[0].text


def query_openai(prompt, max_tokens=500):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-5.4", max_completion_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def parse_json_response(text):
    import re
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    nums = re.findall(r"[\d.]+", text)
    if len(nums) >= 5:
        try:
            return [float(n) for n in nums]
        except ValueError:
            pass
    return None


def run_autointerpretability(model_name, metadata):
    """Run autointerpretability for one model's protein-level features."""
    log.info(f"\n{'='*60}")
    log.info(f"AUTOINTERPRETABILITY: {model_name}")
    log.info(f"{'='*60}")

    import h5py
    feat_dir = FEATURE_ROOT / model_name / "protein"
    with h5py.File(feat_dir / "features.h5", "r") as f:
        Z = f["features"][:]  # (n_proteins, d_sae)
        protein_ids = [s.decode() for s in f["protein_ids"][:]]

    # Find active features
    active_mask = (Z > 0).any(axis=0)
    active_features = np.where(active_mask)[0]
    log.info(f"  {len(protein_ids)} proteins, {len(active_features)} active features")

    # Sample features if too many
    rng = np.random.RandomState(42)
    if len(active_features) > MAX_FEATURES:
        features_to_test = rng.choice(active_features, MAX_FEATURES, replace=False)
        log.info(f"  Sampling {MAX_FEATURES} features for API cost control")
    else:
        features_to_test = active_features

    # Split describe/validate
    describe_ids = protein_ids[:N_DESCRIBE]
    validate_ids = protein_ids[N_DESCRIBE:]
    log.info(f"  Describe: {len(describe_ids)}, Validate: {len(validate_ids)}")

    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    results = []
    for fi, fid in enumerate(features_to_test):
        fid = int(fid)

        # Rank describe proteins by activation
        desc_acts = [(pid, float(Z[id_to_idx[pid], fid])) for pid in describe_ids]
        desc_acts.sort(key=lambda x: -x[1])

        global_max = max(Z[:, fid].max(), 1e-8)

        # Build description prompt
        top_active = desc_acts[:N_EXAMPLE_ACTIVE]
        bottom_inactive = desc_acts[-N_EXAMPLE_INACTIVE:]

        active_list = ""
        for i, (pid, act) in enumerate(top_active):
            meta = metadata.get(pid, {})
            name = meta.get("protein_name", pid)[:80]
            org = meta.get("organism", "")[:40]
            pfam = ", ".join(meta.get("pfam", [])[:3]) or "N/A"
            go_terms = [g.get("term", "").split(":")[-1].strip()
                        for g in meta.get("go_terms", [])[:4]]
            go_str = "; ".join(go_terms) or "N/A"
            active_list += (f"  {i+1}. {name} ({pid}, {org})\n"
                           f"     Act: {act/global_max:.3f} | Pfam: {pfam} | GO: {go_str}\n")

        inactive_list = ""
        for i, (pid, act) in enumerate(bottom_inactive):
            meta = metadata.get(pid, {})
            name = meta.get("protein_name", pid)[:80]
            org = meta.get("organism", "")[:40]
            inactive_list += f"  {i+1}. {name} ({pid}, {org}) Act: {act/global_max:.3f}\n"

        desc_prompt = f"""You are an expert protein biochemist. A sparse autoencoder feature activates on certain proteins. Based on the proteins below, interpret what this feature detects.

## Highly activating proteins:
{active_list}
## Low/non-activating proteins:
{inactive_list}

What biological pattern does this feature detect? Focus on what active proteins share that inactive ones lack.

Respond in JSON: {{"description": "...", "category": "protein_family|function|localization|structure|metabolism|unknown", "confidence": "high|medium|low"}}"""

        try:
            desc_resp = query_claude(desc_prompt)
            desc_parsed = parse_json_response(desc_resp)
            if desc_parsed and isinstance(desc_parsed, dict):
                description = desc_parsed.get("description", desc_resp[:500])
                category = desc_parsed.get("category", "unknown")
            else:
                description = desc_resp[:500]
                category = "unknown"
        except Exception as e:
            description = f"Error: {e}"
            category = "unknown"

        # Validate on held-out proteins
        val_acts = [(pid, float(Z[id_to_idx[pid], fid])) for pid in validate_ids]
        val_acts.sort(key=lambda x: -x[1])

        # Select diverse range for validation
        n = len(val_acts)
        top = val_acts[:8]
        mid = val_acts[n//3:n//3+7]
        bottom = val_acts[-8:]
        extra_idx = rng.choice(range(15, n-15), min(7, max(0, n-30)), replace=False) if n > 30 else []
        extra = [val_acts[i] for i in extra_idx]
        val_selected = list({v[0]: v for v in (top + mid + bottom + extra)}.values())[:N_VALIDATE]

        actual_values = [v[1] / global_max for v in val_selected]

        val_protein_list = ""
        for j, (pid, act) in enumerate(val_selected):
            meta = metadata.get(pid, {})
            name = meta.get("protein_name", pid)[:80]
            org = meta.get("organism", "")[:40]
            pfam = ", ".join(meta.get("pfam", [])[:3]) or "N/A"
            val_protein_list += f"  {j+1}. {name} ({pid}, {org}) Pfam: {pfam}\n"

        val_prompt = f"""A protein language model feature has been described as:
"{description}"

For each protein below, predict activation (0.0 = none, 1.0 = maximum).

## Proteins:
{val_protein_list}

Return ONLY a JSON list of {len(val_selected)} numbers: [pred_1, pred_2, ...]"""

        feature_result = {
            "feature_id": fid,
            "description": description,
            "category": category,
            "n_validate": len(val_selected),
            "actual_values": actual_values,
        }

        for model_label, query_fn in [("claude", query_claude), ("gpt", query_openai)]:
            try:
                resp = query_fn(val_prompt)
                parsed = parse_json_response(resp)
                if isinstance(parsed, list) and len(parsed) >= 3:
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
                    feature_result[f"{model_label}_pearson_r"] = float(r)
                    feature_result[f"{model_label}_pearson_p"] = float(p)
                else:
                    feature_result[f"{model_label}_pearson_r"] = None
            except Exception as e:
                log.warning(f"  {model_label} error f/{fid}: {e}")
                feature_result[f"{model_label}_pearson_r"] = None

        results.append(feature_result)

        if (fi + 1) % 20 == 0:
            valid_r = [r["claude_pearson_r"] for r in results
                       if r.get("claude_pearson_r") is not None]
            median_r = np.median(valid_r) if valid_r else float("nan")
            log.info(f"  Feature {fi+1}/{len(features_to_test)} | "
                     f"Claude median r={median_r:.3f} (n={len(valid_r)})")

    # Summary
    for model_label in ["claude", "gpt"]:
        valid = [r[f"{model_label}_pearson_r"] for r in results
                 if r.get(f"{model_label}_pearson_r") is not None
                 and not np.isnan(r[f"{model_label}_pearson_r"])]
        if valid:
            log.info(f"  {model_label.upper()}: median r={np.median(valid):.3f}, "
                     f"mean r={np.mean(valid):.3f} (n={len(valid)})")

    cats = Counter(r["category"] for r in results)
    log.info(f"  Categories: {dict(cats)}")

    return {
        "model": model_name,
        "n_features_tested": len(results),
        "summary": {
            model_label: {
                "median_r": float(np.median(v)) if (v := [r[f"{model_label}_pearson_r"]
                    for r in results if r.get(f"{model_label}_pearson_r") is not None
                    and not np.isnan(r[f"{model_label}_pearson_r"])]) else None,
                "mean_r": float(np.mean(v)) if v else None,
                "n_valid": len(v) if v else 0,
            }
            for model_label in ["claude", "gpt"]
        },
        "per_feature": results,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    results = {}
    for model_name in ["esm3", "esm2"]:
        feat_path = FEATURE_ROOT / model_name / "protein" / "features.h5"
        if not feat_path.exists():
            log.warning(f"No protein features for {model_name}, skipping")
            continue
        results[model_name] = run_autointerpretability(model_name, metadata)

    out_path = OUTPUT_DIR / "autointerpretability.json"

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
        return obj

    with open(out_path, "w") as f:
        json.dump(make_safe(results), f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
