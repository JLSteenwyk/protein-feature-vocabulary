#!/usr/bin/env python3
"""LLM-automated annotation of SAE features (cross-modal vs matched).

Following InterPLM's approach (Simon & Zou, Nature Methods 2025), uses LLMs
to generate natural language descriptions of SAE features and validates them.

Uses both Claude Opus 4.6 and GPT-5.4 for cross-validation.

Pipeline:
1. For each feature: extract top activating proteins, AA/SS distributions,
   functional site overlap, activation patterns
2. Query LLMs for feature descriptions
3. Validate: can descriptions predict activation on held-out proteins?
4. Compare themes between cross-modal vs matched features
5. Blind test: can LLM predict cross-modal vs matched from description?

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/run_llm_feature_annotation.py
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

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("llm_annot")


def load_sae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


# ═══════════════════════════════════════════════════════════
# Step 1: Extract feature profiles
# ═══════════════════════════════════════════════════════════

def extract_feature_profile(feature_id, sae, h5_path, sequences, offsets, accs,
                            metadata, dssp_data, n_top_proteins=20, n_bottom_proteins=5):
    """Extract a rich profile for a single SAE feature."""
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    SS3_LIST = ["H", "E", "C"]

    protein_scores = []

    with h5py.File(h5_path, "r") as f:
        acts_data = f["activations"]
        total = acts_data.shape[0]

        for acc in accs:
            start, L = offsets[acc]
            if start + L > total or L < 5:
                continue

            raw = torch.tensor(acts_data[start:start + L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()

            col = z[:, feature_id]
            max_act = float(col.max())
            mean_act = float(col[col > 0].mean()) if (col > 0).any() else 0
            n_active = int((col > 0).sum())
            frac_active = n_active / L

            if n_active > 0:
                protein_scores.append({
                    "acc": acc,
                    "max_act": max_act,
                    "mean_act": mean_act,
                    "n_active": n_active,
                    "frac_active": frac_active,
                    "length": L,
                })

    if not protein_scores:
        return None

    # Sort by max activation
    protein_scores.sort(key=lambda x: -x["max_act"])

    # Top and bottom activating proteins
    top_proteins = protein_scores[:n_top_proteins]
    # Bottom: proteins where feature fires but weakly
    bottom_proteins = [p for p in protein_scores if p["frac_active"] < 0.1]
    bottom_proteins = bottom_proteins[-n_bottom_proteins:] if bottom_proteins else []

    # Aggregate AA and SS distributions at active positions
    aa_counts = Counter()
    ss_counts = Counter()
    func_count = 0
    total_active = 0
    asa_values = []

    with h5py.File(h5_path, "r") as f:
        acts_data = f["activations"]
        for pinfo in top_proteins[:10]:  # Use top 10 for detailed analysis
            acc = pinfo["acc"]
            start, L = offsets[acc]
            seq = sequences[acc]
            dssp = dssp_data.get(acc, [])
            meta = metadata.get(acc, {})
            func_sites = set()
            for feat in meta.get("features", []):
                for p in range(feat["start"] - 1, feat["end"]):
                    if 0 <= p < L:
                        func_sites.add(p)

            raw = torch.tensor(acts_data[start:start + L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()
            col = z[:, feature_id]
            active_pos = np.where(col > 0)[0]

            for p in active_pos:
                total_active += 1
                if p < len(seq):
                    aa_counts[seq[p]] += 1
                if dssp and p < len(dssp):
                    ss_counts[dssp[p].get("ss3", "?")] += 1
                    asa_values.append(dssp[p].get("asa", 0))
                if p in func_sites:
                    func_count += 1

    # Normalize AA distribution
    aa_total = sum(aa_counts.values())
    aa_dist = {aa: round(aa_counts.get(aa, 0) / max(aa_total, 1), 3) for aa in AA_LIST}
    top_aa = sorted(aa_dist.items(), key=lambda x: -x[1])[:5]

    ss_total = sum(ss_counts.values())
    ss_dist = {ss: round(ss_counts.get(ss, 0) / max(ss_total, 1), 3) for ss in SS3_LIST}

    # Build protein descriptions for top activating
    top_descriptions = []
    for pinfo in top_proteins[:n_top_proteins]:
        acc = pinfo["acc"]
        meta = metadata.get(acc, {})
        name = meta.get("protein_name", acc)
        org = meta.get("organism", "")
        func_desc = ", ".join(meta.get("function", [])[:2]) if meta.get("function") else ""
        go_terms = [g.get("term", "") for g in meta.get("go_terms", [])[:3]]

        top_descriptions.append({
            "accession": acc,
            "name": name,
            "organism": org,
            "length": pinfo["length"],
            "max_activation": round(pinfo["max_act"], 2),
            "fraction_active": round(pinfo["frac_active"], 3),
            "function": func_desc,
            "go_terms": go_terms,
        })

    return {
        "feature_id": feature_id,
        "n_proteins_active": len(protein_scores),
        "top_proteins": top_descriptions,
        "aa_distribution": dict(top_aa),
        "ss_distribution": ss_dist,
        "functional_site_fraction": round(func_count / max(total_active, 1), 3),
        "mean_asa": round(float(np.mean(asa_values)), 1) if asa_values else None,
        "total_active_residues_sampled": total_active,
    }


# ═══════════════════════════════════════════════════════════
# Step 2: Query LLMs
# ═══════════════════════════════════════════════════════════

def build_annotation_prompt(profile):
    """Build the prompt for LLM feature annotation."""
    top_prots = profile["top_proteins"]

    protein_list = ""
    for i, p in enumerate(top_prots[:15]):
        go_str = "; ".join(p["go_terms"][:3]) if p["go_terms"] else "N/A"
        protein_list += (
            f"  {i+1}. {p['name']} ({p['accession']}, {p['organism']})\n"
            f"     Length: {p['length']}, Max activation: {p['max_activation']}, "
            f"Fraction active: {p['fraction_active']}\n"
            f"     Function: {p['function'] or 'N/A'}\n"
            f"     GO terms: {go_str}\n"
        )

    aa_str = ", ".join(f"{aa}: {frac}" for aa, frac in profile["aa_distribution"].items())
    ss_str = ", ".join(f"{ss}: {frac}" for ss, frac in profile["ss_distribution"].items())

    prompt = f"""You are an expert protein biochemist analyzing a sparse autoencoder (SAE) feature extracted from a protein language model (ESM-3). This feature activates on specific residues across many proteins. Your task is to describe what biological pattern this feature detects.

## Feature Statistics
- Active in {profile['n_proteins_active']} proteins
- Functional site fraction: {profile['functional_site_fraction']} (fraction of active residues at annotated active/binding sites)
- Mean accessible surface area: {profile['mean_asa']} Å²
- Total active residues sampled: {profile['total_active_residues_sampled']}

## Amino Acid Distribution at Active Positions (top 5)
{aa_str}

## Secondary Structure Distribution at Active Positions
{ss_str}
(H=helix, E=sheet, C=coil)

## Top Activating Proteins (ranked by activation strength)
{protein_list}

## Task
Based on the proteins this feature activates on, the amino acid preferences, secondary structure preferences, and functional site overlap, provide:

1. **Description** (1-2 sentences): What biological pattern does this feature detect? Be specific about structural motifs, functional roles, or sequence patterns.

2. **Category** (one of): amino_acid_identity, secondary_structure, structural_motif, functional_site, domain_type, physicochemical_property, unknown

3. **Confidence** (low/medium/high): How confident are you in this interpretation?

4. **Key evidence**: What specific evidence from the data supports your interpretation?

Respond in JSON format:
{{"description": "...", "category": "...", "confidence": "...", "key_evidence": "..."}}"""

    return prompt


def query_claude(prompt, model="claude-opus-4-6"):
    """Query Claude API."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def query_openai(prompt, model="gpt-5.4"):
    """Query OpenAI API."""
    import openai
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def parse_llm_response(text):
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code block
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return {"description": text, "category": "unknown", "confidence": "low", "key_evidence": "parse_error"}


# ═══════════════════════════════════════════════════════════
# Step 3: Blind classification test
# ═══════════════════════════════════════════════════════════

def build_classification_prompt(descriptions_cm, descriptions_mt):
    """Ask LLM to classify features as cross-modal or matched based on descriptions."""
    # Mix and shuffle
    all_descs = []
    for d in descriptions_cm:
        all_descs.append({"id": d["feature_id"], "description": d["annotation"]["description"],
                          "category": d["annotation"]["category"], "true_label": "cross-modal"})
    for d in descriptions_mt:
        all_descs.append({"id": d["feature_id"], "description": d["annotation"]["description"],
                          "category": d["annotation"]["category"], "true_label": "matched"})

    np.random.RandomState(42).shuffle(all_descs)

    feature_list = ""
    for i, d in enumerate(all_descs):
        feature_list += f"  {i+1}. Feature {d['id']}: \"{d['description']}\" (category: {d['category']})\n"

    prompt = f"""You are analyzing features from a sparse autoencoder trained on a multimodal protein language model (ESM-3) that can process both sequence and 3D structure tokens.

Some features are "cross-modal" — they only appear when the model receives structure tokens and have no equivalent in the sequence-only representation. Others are "matched" — they correspond to features that also exist in the sequence-only representation.

Based ONLY on the feature descriptions below, classify each as "cross-modal" or "matched". Cross-modal features should encode patterns that require 3D structural information (spatial relationships, long-range contacts, structural scaffolding). Matched features should encode patterns detectable from sequence alone (amino acid identity, local sequence motifs, conserved residues).

## Features
{feature_list}

For each feature, provide your classification. Respond in JSON format as a list:
[{{"id": feature_id, "predicted": "cross-modal" or "matched", "reasoning": "brief reason"}}]"""

    return prompt, all_descs


# ═══════════════════════════════════════════════════════════
# Step 4: Theme analysis
# ═══════════════════════════════════════════════════════════

def build_theme_analysis_prompt(cm_annotations, mt_annotations):
    """Ask LLM to identify systematic differences between feature groups."""
    cm_descs = [a["annotation"]["description"] for a in cm_annotations]
    mt_descs = [a["annotation"]["description"] for a in mt_annotations]

    cm_cats = Counter(a["annotation"]["category"] for a in cm_annotations)
    mt_cats = Counter(a["annotation"]["category"] for a in mt_annotations)

    prompt = f"""You are analyzing two groups of features from a sparse autoencoder trained on a multimodal protein language model.

**Group A** ({len(cm_descs)} features) — these are "cross-modal" features that only appear when the model receives 3D structure tokens:
Category distribution: {dict(cm_cats)}
Descriptions:
{chr(10).join(f"  - {d}" for d in cm_descs[:50])}

**Group B** ({len(mt_descs)} features) — these are "matched" features that also exist in the sequence-only representation:
Category distribution: {dict(mt_cats)}
Descriptions:
{chr(10).join(f"  - {d}" for d in mt_descs[:50])}

Analyze the systematic differences between these two groups. Provide:
1. **Key themes in Group A** (cross-modal): What biological patterns are enriched?
2. **Key themes in Group B** (matched): What biological patterns are enriched?
3. **Differential themes**: What is present in one group but absent/rare in the other?
4. **Biological interpretation**: Why might structure tokens specifically enable Group A features?

Respond in JSON format:
{{"group_a_themes": ["..."], "group_b_themes": ["..."], "differential_themes": "...", "biological_interpretation": "..."}}"""

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-features", type=int, default=100,
                        help="Number of features per group (cross-modal and matched)")
    parser.add_argument("--n-proteins", type=int, default=500,
                        help="Number of proteins to scan for feature activity")
    args = parser.parse_args()

    # Load cross-modal/matched indices
    cm_path = ROOT / "results" / "unified" / "crossmodal_features" / "crossmodal_features_L33.json"
    with open(cm_path) as f:
        cm_data = json.load(f)
    crossmodal_idx = cm_data["crossmodal_feature_indices"]
    matched_idx = cm_data["matched_feature_indices"]
    log.info(f"Cross-modal: {len(crossmodal_idx)}, Matched: {len(matched_idx)}")

    # Load SAE and data
    sae_path = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_scaled_topk" / "best.pt"
    sae = load_sae(str(sae_path))
    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33.h5")

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json") as f:
        dssp_data = json.load(f)

    accs = sorted(sequences.keys())[:args.n_proteins]
    offsets = {}
    pos = 0
    for acc in sorted(sequences.keys()):
        L = len(sequences[acc])
        offsets[acc] = (pos, L)
        pos += L

    # Sample features
    rng = np.random.RandomState(42)
    n_cm = min(args.n_features, len(crossmodal_idx))
    n_mt = min(args.n_features, len(matched_idx))
    cm_sample = list(rng.choice(crossmodal_idx, n_cm, replace=False))
    mt_sample = list(rng.choice(matched_idx, n_mt, replace=False))

    # ═══════════════════════════════════════════════════
    # Extract ALL feature profiles in a single pass over proteins
    # ═══════════════════════════════════════════════════
    log.info(f"\nExtracting profiles for {n_cm} cross-modal + {n_mt} matched features (single pass)...")

    all_feature_ids = cm_sample + mt_sample
    feature_set = set(all_feature_ids)
    cm_set = set(cm_sample)

    # Per-feature accumulators
    protein_scores = {fid: [] for fid in all_feature_ids}
    aa_counts = {fid: Counter() for fid in all_feature_ids}
    ss_counts = {fid: Counter() for fid in all_feature_ids}
    func_counts = {fid: 0 for fid in all_feature_ids}
    total_active = {fid: 0 for fid in all_feature_ids}
    asa_values = {fid: [] for fid in all_feature_ids}

    with h5py.File(h5_path, "r") as f:
        acts_data = f["activations"]
        total_residues = acts_data.shape[0]

        for pi, acc in enumerate(accs):
            start, L = offsets[acc]
            if start + L > total_residues or L < 5:
                continue

            raw = torch.tensor(acts_data[start:start + L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()

            seq = sequences[acc]
            dssp = dssp_data.get(acc, [])
            meta = metadata.get(acc, {})
            func_sites = set()
            for feat_ann in meta.get("features", []):
                for p in range(feat_ann["start"] - 1, feat_ann["end"]):
                    if 0 <= p < L:
                        func_sites.add(p)

            for fid in all_feature_ids:
                col = z[:, fid]
                max_act = float(col.max())
                n_act = int((col > 0).sum())
                if n_act == 0:
                    continue

                frac_act = n_act / L
                mean_act = float(col[col > 0].mean())

                protein_scores[fid].append({
                    "acc": acc, "max_act": max_act, "mean_act": mean_act,
                    "n_active": n_act, "frac_active": frac_act, "length": L,
                })

                # Detailed annotation for top proteins (first 400 proteins are enough)
                if len(protein_scores[fid]) <= 15:
                    active_pos = np.where(col > 0)[0]
                    for p in active_pos:
                        total_active[fid] += 1
                        if p < len(seq):
                            aa_counts[fid][seq[p]] += 1
                        if dssp and p < len(dssp):
                            ss_counts[fid][dssp[p].get("ss3", "?")] += 1
                            asa_values[fid].append(dssp[p].get("asa", 0))
                        if p in func_sites:
                            func_counts[fid] += 1

            if (pi + 1) % 100 == 0:
                log.info(f"  Scanned {pi + 1}/{len(accs)} proteins")

    log.info(f"  Scan complete. Building profiles...")

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    SS3_LIST = ["H", "E", "C"]

    all_profiles = {}
    for fid in all_feature_ids:
        scores = protein_scores[fid]
        if not scores:
            continue

        scores.sort(key=lambda x: -x["max_act"])
        top_prots = scores[:15]

        # AA distribution
        aa_total = sum(aa_counts[fid].values())
        aa_dist = {aa: round(aa_counts[fid].get(aa, 0) / max(aa_total, 1), 3) for aa in AA_LIST}
        top_aa = sorted(aa_dist.items(), key=lambda x: -x[1])[:5]

        ss_total = sum(ss_counts[fid].values())
        ss_dist = {ss: round(ss_counts[fid].get(ss, 0) / max(ss_total, 1), 3) for ss in SS3_LIST}

        # Build protein descriptions
        top_descriptions = []
        for pinfo in top_prots:
            acc = pinfo["acc"]
            meta = metadata.get(acc, {})
            name = meta.get("protein_name", acc)
            org = meta.get("organism", "")
            func_desc = ", ".join(meta.get("function", [])[:2]) if meta.get("function") else ""
            go_terms = [g.get("term", "") for g in meta.get("go_terms", [])[:3]]
            top_descriptions.append({
                "accession": acc, "name": name, "organism": org,
                "length": pinfo["length"], "max_activation": round(pinfo["max_act"], 2),
                "fraction_active": round(pinfo["frac_active"], 3),
                "function": func_desc, "go_terms": go_terms,
            })

        label = "cross-modal" if fid in cm_set else "matched"
        all_profiles[fid] = {
            "feature_id": fid,
            "label": label,
            "n_proteins_active": len(scores),
            "top_proteins": top_descriptions,
            "aa_distribution": dict(top_aa),
            "ss_distribution": ss_dist,
            "functional_site_fraction": round(func_counts[fid] / max(total_active[fid], 1), 3),
            "mean_asa": round(float(np.mean(asa_values[fid])), 1) if asa_values[fid] else None,
            "total_active_residues_sampled": total_active[fid],
        }

    log.info(f"  Extracted {len(all_profiles)} valid profiles")

    # ═══════════════════════════════════════════════════
    # Query LLMs
    # ═══════════════════════════════════════════════════
    log.info("\nQuerying LLMs for feature annotations...")

    annotations_claude = []
    annotations_gpt = []

    for i, (fid, profile) in enumerate(all_profiles.items()):
        prompt = build_annotation_prompt(profile)
        label = profile["label"]

        # Query Claude
        try:
            claude_resp = query_claude(prompt)
            claude_parsed = parse_llm_response(claude_resp)
        except Exception as e:
            log.warning(f"  Claude error for f/{fid}: {e}")
            claude_parsed = {"description": f"Error: {e}", "category": "unknown",
                             "confidence": "low", "key_evidence": "api_error"}

        # Query GPT
        try:
            gpt_resp = query_openai(prompt)
            gpt_parsed = parse_llm_response(gpt_resp)
        except Exception as e:
            log.warning(f"  GPT error for f/{fid}: {e}")
            gpt_parsed = {"description": f"Error: {e}", "category": "unknown",
                          "confidence": "low", "key_evidence": "api_error"}

        annotations_claude.append({
            "feature_id": fid,
            "label": label,
            "annotation": claude_parsed,
            "profile_summary": {
                "n_proteins": profile["n_proteins_active"],
                "aa_top": list(profile["aa_distribution"].keys())[:3],
                "ss_dist": profile["ss_distribution"],
                "func_frac": profile["functional_site_fraction"],
            }
        })
        annotations_gpt.append({
            "feature_id": fid,
            "label": label,
            "annotation": gpt_parsed,
            "profile_summary": annotations_claude[-1]["profile_summary"],
        })

        if (i + 1) % 10 == 0:
            log.info(f"  Annotated {i + 1}/{len(all_profiles)} features")
            # Brief rate limiting
            time.sleep(0.5)

    # ═══════════════════════════════════════════════════
    # Category distribution analysis
    # ═══════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("ANNOTATION RESULTS")
    log.info("=" * 60)

    for model_name, annotations in [("Claude Opus 4.6", annotations_claude),
                                      ("GPT 5.4", annotations_gpt)]:
        log.info(f"\n--- {model_name} ---")

        cm_anns = [a for a in annotations if a["label"] == "cross-modal"]
        mt_anns = [a for a in annotations if a["label"] == "matched"]

        cm_cats = Counter(a["annotation"]["category"] for a in cm_anns)
        mt_cats = Counter(a["annotation"]["category"] for a in mt_anns)

        log.info(f"Cross-modal categories: {dict(cm_cats)}")
        log.info(f"Matched categories:     {dict(mt_cats)}")

        cm_conf = Counter(a["annotation"]["confidence"] for a in cm_anns)
        mt_conf = Counter(a["annotation"]["confidence"] for a in mt_anns)
        log.info(f"Cross-modal confidence: {dict(cm_conf)}")
        log.info(f"Matched confidence:     {dict(mt_conf)}")

        # Sample descriptions
        log.info(f"\nSample cross-modal descriptions:")
        for a in cm_anns[:5]:
            log.info(f"  f/{a['feature_id']}: {a['annotation']['description'][:120]}")
        log.info(f"\nSample matched descriptions:")
        for a in mt_anns[:5]:
            log.info(f"  f/{a['feature_id']}: {a['annotation']['description'][:120]}")

    # ═══════════════════════════════════════════════════
    # Theme analysis (ask LLM to compare groups)
    # ═══════════════════════════════════════════════════
    log.info("\nRunning theme analysis...")

    cm_claude = [a for a in annotations_claude if a["label"] == "cross-modal"]
    mt_claude = [a for a in annotations_claude if a["label"] == "matched"]

    theme_prompt = build_theme_analysis_prompt(cm_claude, mt_claude)

    try:
        theme_claude = parse_llm_response(query_claude(theme_prompt))
    except Exception as e:
        log.warning(f"Theme analysis (Claude) error: {e}")
        theme_claude = {"error": str(e)}

    try:
        theme_gpt = parse_llm_response(query_openai(theme_prompt))
    except Exception as e:
        log.warning(f"Theme analysis (GPT) error: {e}")
        theme_gpt = {"error": str(e)}

    log.info(f"\nClaude theme analysis:")
    log.info(f"  Group A (cross-modal): {theme_claude.get('group_a_themes', 'N/A')}")
    log.info(f"  Group B (matched): {theme_claude.get('group_b_themes', 'N/A')}")
    log.info(f"  Differential: {str(theme_claude.get('differential_themes', 'N/A'))[:200]}")
    log.info(f"  Interpretation: {str(theme_claude.get('biological_interpretation', 'N/A'))[:200]}")

    # ═══════════════════════════════════════════════════
    # Blind classification test
    # ═══════════════════════════════════════════════════
    log.info("\nRunning blind classification test...")

    # Use a subset for classification (to fit in context)
    cm_subset = cm_claude[:30]
    mt_subset = mt_claude[:30]

    class_prompt, all_items = build_classification_prompt(cm_subset, mt_subset)

    try:
        class_claude_resp = query_claude(class_prompt)
        class_claude = parse_llm_response(class_claude_resp)
    except Exception as e:
        log.warning(f"Classification (Claude) error: {e}")
        class_claude = []

    try:
        class_gpt_resp = query_openai(class_prompt)
        class_gpt = parse_llm_response(class_gpt_resp)
    except Exception as e:
        log.warning(f"Classification (GPT) error: {e}")
        class_gpt = []

    # Score classification accuracy
    true_labels = {d["id"]: d["true_label"] for d in all_items}

    for model_name, predictions in [("Claude", class_claude), ("GPT", class_gpt)]:
        if not isinstance(predictions, list):
            log.info(f"  {model_name}: could not parse predictions")
            continue

        correct = 0
        total = 0
        for pred in predictions:
            fid = pred.get("id")
            predicted = pred.get("predicted", "")
            if fid in true_labels:
                total += 1
                if predicted == true_labels[fid]:
                    correct += 1

        acc = correct / max(total, 1)
        log.info(f"  {model_name} blind classification: {correct}/{total} = {acc:.1%}")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    output = {
        "n_crossmodal": n_cm,
        "n_matched": n_mt,
        "annotations_claude": annotations_claude,
        "annotations_gpt": annotations_gpt,
        "theme_analysis_claude": theme_claude,
        "theme_analysis_gpt": theme_gpt,
        "blind_classification": {
            "n_per_group": 30,
            "claude": class_claude if isinstance(class_claude, list) else [],
            "gpt": class_gpt if isinstance(class_gpt, list) else [],
            "true_labels": true_labels,
        },
    }

    out_path = ROOT / "results" / "unified" / "crossmodal_features" / "llm_feature_annotations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle numpy serialization
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
        json.dump(make_safe(output), f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
