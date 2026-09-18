#!/usr/bin/env python3
"""Novel SAE feature characterization for ESM-2 and ESM-3.

Identifies uncharacterized SAE features (not matching known biological
annotations) and characterizes them by:
  1. Sequence context analysis (surrounding amino acid composition)
  2. Physicochemical properties (hydrophobicity, charge, flexibility)
  3. Structural context (secondary structure, solvent accessibility)
  4. Activation statistics (sparsity, magnitude, position bias)
  5. Co-activation clustering (features that fire together)

The goal is to generate testable hypotheses about what these features encode.

Run as:
    ./env/bin/python scripts/unified/run_novel_feature_characterization.py --model esm2 --device cuda:0
    ./env/bin/python scripts/unified/run_novel_feature_characterization.py --model esm3 --device cuda:1

Output: results/unified/{model}/novel_feature_characterization.json
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
import numpy as np
from scipy import stats as scipy_stats


def setup_logging(model_name):
    out_dir = ROOT / "results" / "unified" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "novel_feature_characterization_log.txt", mode='w'),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("novel"), out_dir


# Physicochemical properties per amino acid
HYDROPHOBICITY = {  # Kyte-Doolittle scale
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
    'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
    'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5,
}
CHARGE = {
    'K': 1, 'R': 1, 'H': 0.5, 'D': -1, 'E': -1,
    'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0,
    'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0,
}
FLEXIBILITY = {  # B-factor proxy (Vihinen & Mantsala, 1989)
    'G': 1.0, 'S': 0.9, 'D': 0.85, 'N': 0.85, 'P': 0.8, 'K': 0.75,
    'E': 0.7, 'Q': 0.65, 'T': 0.6, 'R': 0.55, 'A': 0.5, 'H': 0.45,
    'M': 0.4, 'C': 0.35, 'L': 0.3, 'V': 0.25, 'I': 0.2, 'F': 0.15,
    'Y': 0.1, 'W': 0.05,
}
STANDARD_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def get_physicochemical(aa):
    """Get physicochemical properties for an amino acid."""
    return {
        'hydrophobicity': HYDROPHOBICITY.get(aa, 0),
        'charge': CHARGE.get(aa, 0),
        'flexibility': FLEXIBILITY.get(aa, 0.5),
    }


def load_sae(model_name, layer, condition="S"):
    """Load a trained SAE checkpoint."""
    from sae.model import build_sae
    sae_dir = ROOT / "models" / "sae"
    cond = condition.replace("+", "_")

    if model_name == "esm2":
        for prefix in ["esm2_scaled", "esm2"]:
            for fname in ["best.pt", "final.pt"]:
                candidate = sae_dir / prefix / f"layer_{layer}_topk" / fname
                if candidate.exists():
                    ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
                    sae = build_sae(ckpt["sae_config"])
                    sae.load_state_dict(ckpt["model_state_dict"])
                    sae.eval()
                    return sae
    else:
        for prefix in ["esm3_scaled", "esm3"]:
            for fname in ["best.pt", "final.pt"]:
                candidate = sae_dir / prefix / f"{cond}_layer_{layer}_topk" / fname
                if candidate.exists():
                    ckpt = torch.load(candidate, map_location="cpu", weights_only=False)
                    sae = build_sae(ckpt["sae_config"])
                    sae.load_state_dict(ckpt["model_state_dict"])
                    sae.eval()
                    return sae
    return None


def classify_feature(aa_counts, ss_counts, func_count, total_count):
    """Classify a feature as AA-specific, SS-specific, functional, or uncharacterized."""
    if total_count < 10:
        return "dead"

    # Check AA specificity: >50% activations on one AA
    if aa_counts:
        top_aa, top_count = aa_counts.most_common(1)[0]
        if top_count / total_count > 0.5:
            return f"aa_{top_aa}"

    # Check SS specificity: >60% activations on one SS type
    if ss_counts:
        top_ss, top_count = ss_counts.most_common(1)[0]
        if top_count / total_count > 0.6:
            return f"ss_{top_ss}"

    # Check functional site enrichment: >3x background rate
    if func_count / max(total_count, 1) > 0.15:  # ~5% background
        return "functional"

    return "uncharacterized"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-proteins", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=400)
    args = parser.parse_args()

    log, out_dir = setup_logging(args.model)
    log.info(f"=== Novel Feature Characterization: {args.model} ===")
    device = args.device

    # Load data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        all_seqs = json.load(f)
    seqs = {k: v for k, v in all_seqs.items() if len(v) <= args.max_length}
    sorted_accs = sorted(seqs.keys())[:args.max_proteins]
    seqs = {k: seqs[k] for k in sorted_accs}

    # Load DSSP annotations
    dssp_file = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"
    dssp = {}
    if dssp_file.exists():
        with open(dssp_file) as f:
            dssp = json.load(f)

    # Load functional site annotations
    meta_file = ROOT / "data" / "scaled" / "annotations" / "metadata.json"
    functional_sites = {}
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        for acc, data in meta.items():
            sites = set()
            for feat in data.get("features", []):
                if feat.get("type") in ("Active site", "Binding site",
                                         "Metal binding", "Site"):
                    for pos in range(feat["start"], feat["end"] + 1):
                        sites.add(pos - 1)
            if sites:
                functional_sites[acc] = sites

    log.info(f"Loaded {len(seqs)} proteins, {len(dssp)} with DSSP, "
             f"{len(functional_sites)} with functional sites")

    # Select SAE layer
    if args.model == "esm2":
        sae_layer = 24
    else:
        sae_layer = 33

    # Load SAE
    sae = load_sae(args.model, sae_layer, "S")
    if sae is None:
        log.error(f"No SAE found at layer {sae_layer}")
        return
    sae = sae.to(device)
    dict_size = sae.config.dict_size
    log.info(f"Loaded SAE at layer {sae_layer}: {dict_size} features")

    # Load model
    if args.model == "esm2":
        from models.esm2_hooks import load_esm2, ESM2HookManager
        model, tokenizer = load_esm2(device=device)
        hook_mgr = ESM2HookManager(model, layers=[sae_layer], extract_attention=False)
    else:
        from models.esm3_hooks import load_esm3, ESM3HookManager
        model, tokenizers = load_esm3(device=device)
        hook_mgr = ESM3HookManager(model, layers=[sae_layer], extract_attention=False)

    # ================================================================
    # Pass 1: Collect per-feature activation statistics
    # ================================================================
    log.info("Pass 1: Collecting feature activation statistics...")

    # Per-feature collectors
    feat_aa_counts = [Counter() for _ in range(dict_size)]
    feat_ss_counts = [Counter() for _ in range(dict_size)]
    feat_func_counts = np.zeros(dict_size, dtype=int)
    feat_total_counts = np.zeros(dict_size, dtype=int)
    feat_activation_sums = np.zeros(dict_size)
    feat_hydro_sums = np.zeros(dict_size)
    feat_charge_sums = np.zeros(dict_size)
    feat_flex_sums = np.zeros(dict_size)
    feat_rel_pos_sums = np.zeros(dict_size)  # relative position in protein
    feat_asa_sums = np.zeros(dict_size)
    feat_asa_counts = np.zeros(dict_size)

    # Co-activation matrix (sparse — only track top features)
    coact_matrix = np.zeros((min(dict_size, 512), min(dict_size, 512)))

    t0 = time.time()
    n_done = 0

    for acc in sorted_accs:
        seq = seqs[acc]
        L = len(seq)

        try:
            hook_mgr.cache.clear()
            hook_mgr.register()
            if args.model == "esm2":
                inputs = tokenizer(seq, return_tensors="pt",
                                   truncation=True, max_length=1024).to(device)
                with torch.no_grad():
                    model(**inputs)
            else:
                seq_tokens = tokenizers.sequence.encode(seq)
                seq_tensor = torch.tensor(seq_tokens, dtype=torch.long,
                                          device=device).unsqueeze(0)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(sequence_tokens=seq_tensor)
            hook_mgr.remove()

            h = hook_mgr.cache.residual_stream[sae_layer][0][1:-1].float()
            with torch.no_grad():
                z = sae.encode(h[:L].to(device))  # (L, dict_size)
            z_np = z.cpu().numpy()

            # Get DSSP data for this protein
            dssp_data = dssp.get(acc, [])
            func_set = functional_sites.get(acc, set())

            # Collect per-feature stats
            for pos in range(L):
                aa = seq[pos]
                if aa not in STANDARD_AAS:
                    continue

                active_features = np.where(z_np[pos] > 0)[0]
                for fi in active_features:
                    act_val = z_np[pos, fi]
                    feat_aa_counts[fi][aa] += 1
                    feat_total_counts[fi] += 1
                    feat_activation_sums[fi] += act_val

                    # SS
                    if pos < len(dssp_data):
                        ss = dssp_data[pos].get('ss3', 'C')
                        feat_ss_counts[fi][ss] += 1
                        asa = dssp_data[pos].get('asa', -1)
                        if asa >= 0:
                            feat_asa_sums[fi] += asa
                            feat_asa_counts[fi] += 1

                    # Functional
                    if pos in func_set:
                        feat_func_counts[fi] += 1

                    # Physicochemical
                    props = get_physicochemical(aa)
                    feat_hydro_sums[fi] += props['hydrophobicity']
                    feat_charge_sums[fi] += props['charge']
                    feat_flex_sums[fi] += props['flexibility']

                    # Relative position
                    feat_rel_pos_sums[fi] += pos / max(L - 1, 1)

                # Co-activation (top 512 features by total count)
                top_active = [fi for fi in active_features if fi < 512]
                for i in range(len(top_active)):
                    for j in range(i + 1, len(top_active)):
                        coact_matrix[top_active[i], top_active[j]] += 1
                        coact_matrix[top_active[j], top_active[i]] += 1

            n_done += 1
            if n_done % 50 == 0:
                log.info(f"  [{n_done}/{len(sorted_accs)} proteins, {time.time()-t0:.0f}s]")

        except Exception as e:
            log.warning(f"  {acc}: {e}")
            torch.cuda.empty_cache()

    log.info(f"Pass 1 complete: {n_done} proteins, {time.time()-t0:.0f}s")

    # ================================================================
    # Pass 2: Classify features and characterize uncharacterized ones
    # ================================================================
    log.info("Pass 2: Classifying and characterizing features...")

    feature_classes = {}
    uncharacterized = []

    for fi in range(dict_size):
        tc = feat_total_counts[fi]
        category = classify_feature(
            feat_aa_counts[fi], feat_ss_counts[fi],
            feat_func_counts[fi], tc)
        feature_classes[fi] = category

        if category == "uncharacterized":
            uncharacterized.append(fi)

    # Count by category
    class_counts = Counter(feature_classes.values())
    log.info(f"Feature classification:")
    for cls, cnt in class_counts.most_common():
        log.info(f"  {cls}: {cnt}")
    log.info(f"  Uncharacterized: {len(uncharacterized)}")

    # ================================================================
    # Characterize uncharacterized features
    # ================================================================
    log.info(f"Characterizing {len(uncharacterized)} uncharacterized features...")

    novel_features = []
    for fi in uncharacterized:
        tc = feat_total_counts[fi]
        if tc < 20:
            continue

        mean_hydro = feat_hydro_sums[fi] / tc
        mean_charge = feat_charge_sums[fi] / tc
        mean_flex = feat_flex_sums[fi] / tc
        mean_rel_pos = feat_rel_pos_sums[fi] / tc
        mean_act = feat_activation_sums[fi] / tc
        mean_asa = feat_asa_sums[fi] / max(feat_asa_counts[fi], 1)

        # Top AAs
        top_aas = feat_aa_counts[fi].most_common(5)
        aa_entropy = 0
        for aa, cnt in feat_aa_counts[fi].items():
            p = cnt / tc
            aa_entropy -= p * np.log(p + 1e-10)

        # Top SS
        top_ss = feat_ss_counts[fi].most_common(3)

        # Generate hypothesis based on properties
        hypotheses = []
        if mean_hydro > 2.0:
            hypotheses.append("hydrophobic_core")
        elif mean_hydro < -2.0:
            hypotheses.append("charged_surface")
        if abs(mean_charge) > 0.3:
            hypotheses.append("charged_cluster")
        if mean_asa < 20:
            hypotheses.append("buried_residue")
        elif mean_asa > 80:
            hypotheses.append("exposed_surface")
        if mean_rel_pos < 0.15:
            hypotheses.append("n_terminal")
        elif mean_rel_pos > 0.85:
            hypotheses.append("c_terminal")
        if mean_flex > 0.7:
            hypotheses.append("flexible_loop")
        elif mean_flex < 0.3:
            hypotheses.append("rigid_core")
        if aa_entropy < 1.5:
            hypotheses.append("sequence_specific")
        elif aa_entropy > 2.5:
            hypotheses.append("context_dependent")

        # Functional site enrichment (even if below threshold)
        func_frac = feat_func_counts[fi] / tc

        novel_features.append({
            "feature_idx": int(fi),
            "total_activations": int(tc),
            "mean_activation": float(mean_act),
            "aa_entropy": float(aa_entropy),
            "top_aas": [(aa, int(cnt)) for aa, cnt in top_aas],
            "top_ss": [(ss, int(cnt)) for ss, cnt in top_ss],
            "mean_hydrophobicity": float(mean_hydro),
            "mean_charge": float(mean_charge),
            "mean_flexibility": float(mean_flex),
            "mean_relative_position": float(mean_rel_pos),
            "mean_asa": float(mean_asa),
            "functional_site_fraction": float(func_frac),
            "hypotheses": hypotheses,
        })

    # Sort by activation count (most active first)
    novel_features.sort(key=lambda x: -x["total_activations"])
    log.info(f"Characterized {len(novel_features)} novel features (≥20 activations)")

    # Hypothesis distribution
    hyp_counts = Counter()
    for nf in novel_features:
        for h in nf["hypotheses"]:
            hyp_counts[h] += 1
    log.info("Hypothesis distribution:")
    for h, c in hyp_counts.most_common():
        log.info(f"  {h}: {c} features")

    # ================================================================
    # Co-activation clusters among uncharacterized features
    # ================================================================
    log.info("Computing co-activation clusters...")
    unchars_in_range = [fi for fi in uncharacterized if fi < 512 and feat_total_counts[fi] >= 20]
    coact_pairs = []
    for i, fi in enumerate(unchars_in_range):
        for j, fj in enumerate(unchars_in_range):
            if fi < fj:
                coact = coact_matrix[fi, fj]
                # Normalize by geometric mean of counts
                norm = np.sqrt(feat_total_counts[fi] * feat_total_counts[fj])
                if norm > 0:
                    jaccard = coact / norm
                    if jaccard > 0.05:  # significant co-activation
                        coact_pairs.append({
                            "feature_a": int(fi),
                            "feature_b": int(fj),
                            "co_activations": int(coact),
                            "normalized_score": float(jaccard),
                        })

    coact_pairs.sort(key=lambda x: -x["normalized_score"])
    log.info(f"Found {len(coact_pairs)} significant co-activation pairs")

    # ================================================================
    # Summary statistics
    # ================================================================
    # Uncharacterized features by physicochemical profile
    profiles = {
        "hydrophobic_core": [nf for nf in novel_features if "hydrophobic_core" in nf["hypotheses"]],
        "charged_surface": [nf for nf in novel_features if "charged_surface" in nf["hypotheses"]],
        "buried_residue": [nf for nf in novel_features if "buried_residue" in nf["hypotheses"]],
        "exposed_surface": [nf for nf in novel_features if "exposed_surface" in nf["hypotheses"]],
        "flexible_loop": [nf for nf in novel_features if "flexible_loop" in nf["hypotheses"]],
        "terminal": [nf for nf in novel_features
                     if "n_terminal" in nf["hypotheses"] or "c_terminal" in nf["hypotheses"]],
        "sequence_specific": [nf for nf in novel_features if "sequence_specific" in nf["hypotheses"]],
        "context_dependent": [nf for nf in novel_features if "context_dependent" in nf["hypotheses"]],
    }

    summary = {
        "total_features": dict_size,
        "classified": {cls: cnt for cls, cnt in class_counts.most_common()},
        "n_uncharacterized": len(uncharacterized),
        "n_characterized_novel": len(novel_features),
        "hypothesis_counts": dict(hyp_counts.most_common()),
        "profile_counts": {k: len(v) for k, v in profiles.items()},
    }

    output = {
        "model": args.model,
        "sae_layer": sae_layer,
        "n_proteins": n_done,
        "summary": summary,
        "novel_features": novel_features[:200],  # top 200 by activation count
        "co_activation_pairs": coact_pairs[:100],  # top 100 pairs
    }

    out_path = out_dir / "novel_feature_characterization.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
