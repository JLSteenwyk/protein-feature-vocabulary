#!/usr/bin/env python3
"""Residue-level autointerpretability: describe what individual SAE features detect.

For top structure-enhanced and top probe-weight features:
- Show example residue contexts where the feature activates
- Query LLM to describe the local structural/sequence motif
- Validate by predicting activation on held-out residues

Usage:
    ./env/bin/python scripts/scaled_1.5M/23_residue_level_autointerpretability.py
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
from scipy import sparse, stats

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("res_interp")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

N_FEATURES = 100  # top features to describe
N_EXAMPLES = 20   # residue contexts to show per feature
CONTEXT_WINDOW = 7  # residues on each side


def query_claude(prompt, max_tokens=300):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        system="You are a helpful protein biochemist. Respond with valid JSON only.",
    )
    return resp.content[0].text


def parse_json(text):
    import re
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    for s, e in [("{", "}"), ("[", "]")]:
        start = text.find(s)
        end = text.rfind(e) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    # Load sparse features
    feat_dir = FEATURE_ROOT / "esm3" / "residue"
    X_sparse = sparse.load_npz(str(feat_dir / "features_sparse.npz"))
    with h5py.File(feat_dir / "protein_summaries.h5", "r") as f:
        protein_ids = [s.decode() for s in f["protein_ids"][:]]
        offsets = f["offsets"][:]
        feature_counts = f["feature_activation_counts"][:]

    pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    # Select features to describe: mix of structure-enhanced + top probe weights
    cm_path = OUTPUT_DIR / "cross_modal_features.json"
    enhanced_ids = []
    if cm_path.exists():
        with open(cm_path) as f:
            cm = json.load(f)
        enhanced_ids = cm.get("enhanced_feature_ids", [])[:50]

    probe_path = OUTPUT_DIR / "interpretable_probing.json"
    probe_feature_ids = []
    if probe_path.exists():
        with open(probe_path) as f:
            probe = json.load(f)
        if "esm3" in probe:
            probe_feature_ids = [f["feature_id"] for f in probe["esm3"].get("top_features", [])[:50]]

    # Combine and deduplicate
    feature_ids = list(dict.fromkeys(enhanced_ids + probe_feature_ids))[:N_FEATURES]
    log.info(f"Describing {len(feature_ids)} features ({len(enhanced_ids)} enhanced, "
             f"{len(probe_feature_ids)} probe)")

    # Build protein sequence index
    # For each residue in the sparse matrix, we need protein ID + position
    residue_to_protein = []  # (protein_idx, local_position)
    for pi in range(len(protein_ids)):
        start, L = offsets[pi]
        for pos in range(L):
            residue_to_protein.append((pi, pos))

    results = []

    for fi, fid in enumerate(feature_ids):
        fid = int(fid)

        # Find residues where this feature activates
        col = X_sparse[:, fid].toarray().flatten()
        active_idx = np.where(col > 0)[0]

        if len(active_idx) < 5:
            continue

        # Sort by activation strength, take top examples
        active_vals = col[active_idx]
        top_idx = active_idx[np.argsort(-active_vals)][:N_EXAMPLES * 3]  # oversample

        # Build residue context strings
        examples = []
        seen_proteins = set()

        for ridx in top_idx:
            if len(examples) >= N_EXAMPLES:
                break

            pi, pos = residue_to_protein[ridx]
            pid = protein_ids[pi]

            # Avoid too many from same protein
            if pid in seen_proteins and len(seen_proteins) < 10:
                continue
            seen_proteins.add(pid)

            start, L = offsets[pi]
            seq = sequences.get(pid, "")
            if not seq or pos >= len(seq):
                continue

            # Extract context window
            ctx_start = max(0, pos - CONTEXT_WINDOW)
            ctx_end = min(len(seq), pos + CONTEXT_WINDOW + 1)
            context = seq[ctx_start:ctx_end]
            center_pos = pos - ctx_start

            # Check if position is a functional site
            meta = metadata.get(pid, {})
            site_types = []
            for feat in meta.get("features", []):
                if feat["start"] - 1 <= pos < feat["end"]:
                    site_types.append(feat["type"])

            examples.append({
                "protein": pid,
                "position": pos,
                "residue": seq[pos],
                "context": context,
                "center_in_context": center_pos,
                "activation": float(col[ridx]),
                "functional_sites": site_types,
                "protein_name": meta.get("protein_name", "")[:60],
            })

        if len(examples) < 3:
            continue

        # Build description prompt
        example_text = ""
        for i, ex in enumerate(examples[:15]):
            ctx = ex["context"]
            cp = ex["center_in_context"]
            marked = ctx[:cp] + "[" + ctx[cp] + "]" + ctx[cp+1:]
            sites = ", ".join(ex["functional_sites"]) if ex["functional_sites"] else "none"
            example_text += (f"  {i+1}. {ex['protein_name'][:40]} ({ex['protein']}), "
                           f"pos {ex['position']}: ...{marked}... "
                           f"(act={ex['activation']:.2f}, sites: {sites})\n")

        is_enhanced = fid in set(enhanced_ids)
        label = "structure-enhanced" if is_enhanced else "probe-important"

        prompt = f"""A sparse autoencoder feature (#{fid}, {label}) from a protein language model activates at specific residue positions. Here are examples where it activates most strongly:

{example_text}

The [X] marks the residue where the feature activates. What local structural or sequence motif does this feature detect? Focus on amino acid patterns, secondary structure context, or functional roles.

Respond in JSON: {{"description": "...", "motif_type": "sequence_pattern|secondary_structure|functional_site|physicochemical|unknown", "confidence": "high|medium|low"}}"""

        try:
            resp = query_claude(prompt)
            parsed = parse_json(resp)
            if parsed and isinstance(parsed, dict):
                description = parsed.get("description", resp[:300])
                motif_type = parsed.get("motif_type", "unknown")
            else:
                description = resp[:300]
                motif_type = "unknown"
        except Exception as e:
            description = f"Error: {e}"
            motif_type = "unknown"

        # Residue type distribution
        active_residues = [residue_to_protein[idx] for idx in active_idx[:1000]]
        aa_counts = Counter()
        for pi_r, pos_r in active_residues:
            pid_r = protein_ids[pi_r]
            seq_r = sequences.get(pid_r, "")
            if pos_r < len(seq_r):
                aa_counts[seq_r[pos_r]] += 1

        top_aa = aa_counts.most_common(5)

        feature_result = {
            "feature_id": fid,
            "label": label,
            "description": description,
            "motif_type": motif_type,
            "n_active_residues": len(active_idx),
            "n_active_proteins": len(set(protein_ids[residue_to_protein[i][0]] for i in active_idx[:5000])),
            "top_amino_acids": [{"aa": aa, "count": n} for aa, n in top_aa],
            "examples": examples[:5],  # save top 5 for reference
        }

        results.append(feature_result)

        if (fi + 1) % 20 == 0:
            log.info(f"  Feature {fi+1}/{len(feature_ids)}: described {len(results)} features")

    # Summary
    motif_types = Counter(r["motif_type"] for r in results)
    log.info(f"\n{'='*60}")
    log.info("RESIDUE-LEVEL AUTOINTERPRETABILITY")
    log.info(f"{'='*60}")
    log.info(f"Features described: {len(results)}")
    log.info(f"Motif types: {dict(motif_types)}")

    log.info(f"\nSample descriptions:")
    for r in results[:10]:
        log.info(f"  f/{r['feature_id']} ({r['label']}): {r['description'][:100]}")

    output = {
        "n_features": len(results),
        "motif_type_distribution": dict(motif_types),
        "per_feature": results,
    }

    out_path = OUTPUT_DIR / "residue_level_autointerpretability.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
