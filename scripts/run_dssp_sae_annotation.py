#!/usr/bin/env python3
"""DSSP-based structural annotation pipeline for ESM-3 SAE features.

Step 1: Run mkdssp on 199 PDB structures to extract per-residue SS and RSA.
Step 2: Load SAE models, run forward passes, compute structural enrichments.
Step 3: Save enhanced annotations per SAE.
Step 4: Compare S-only vs S+St SAEs for structural feature learning.

All CPU work — no GPU required.
"""

import json
import os
import sys
import subprocess
import re
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import h5py
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY")
STRUCTURES_DIR = BASE / "data" / "pilot" / "structures"
ANNOTATIONS_DIR = BASE / "data" / "pilot" / "annotations"
ACTIVATIONS_DIR = BASE / "data" / "activations" / "esm3_multimodal"
MODELS_DIR = BASE / "models" / "sae" / "esm3"
RESULTS_DIR = BASE / "results" / "phase1" / "modality_saes" / "structural_annotations"
MKDSSP = BASE / "env" / "bin" / "mkdssp"

# Ensure output dirs exist
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add src to path for SAE model loading
sys.path.insert(0, str(BASE / "src"))
from sae.model import SAEConfig, build_sae

# ── Max ASA values (Tien et al., 2013, theoretical) for RSA computation ──────
# Used to convert absolute accessible surface area → relative (0-1)
MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

# Three-letter to one-letter amino acid code
AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Coarse SS mapping: DSSP 8-state → 3-state (H=helix, E=sheet, C=coil)
SS8_TO_SS3 = {
    "H": "H",  # alpha helix
    "G": "H",  # 3-10 helix → helix
    "I": "H",  # pi helix → helix
    "E": "E",  # beta strand
    "B": "E",  # isolated beta bridge → sheet
    "T": "C",  # turn → coil
    "S": "C",  # bend → coil
    " ": "C",  # loop/other → coil
    "C": "C",  # coil
    "P": "C",  # polyproline helix → coil (mkdssp 4.x)
}


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: Run DSSP on all PDB structures
# ═════════════════════════════════════════════════════════════════════════════

def parse_dssp_classic(dssp_output: str):
    """Parse classic DSSP format output to extract per-residue SS and ASA.

    Returns list of dicts with keys: resnum, aa, ss8, ss3, asa, rsa
    """
    residues = []
    in_data = False

    for line in dssp_output.split("\n"):
        # Data starts after the header line containing "#  RESIDUE AA STRUCTURE"
        if "  #  RESIDUE AA STRUCTURE" in line:
            in_data = True
            continue
        if not in_data:
            continue
        if len(line) < 38:
            continue

        # Skip chain breaks (marked with '!' in the AA column)
        aa_char = line[13:14]
        if aa_char == "!" or aa_char == " ":
            continue

        # Parse fields from fixed-width DSSP format
        try:
            resnum = int(line[5:10].strip())
            chain = line[11:12]
            aa = line[13:14]  # one-letter AA code
            ss = line[16:17]  # secondary structure code
            if ss == " ":
                ss = "C"  # map blank to coil
            asa = float(line[35:38].strip())  # accessible surface area
        except (ValueError, IndexError):
            continue

        # Compute RSA from ASA
        # Find the 3-letter code for max ASA lookup
        aa_upper = aa.upper()
        # Map 1-letter to 3-letter for MAX_ASA lookup
        aa1_to_3 = {v: k for k, v in AA3_TO_1.items()}
        aa3 = aa1_to_3.get(aa_upper, None)
        max_asa = MAX_ASA.get(aa3, 200.0)  # fallback
        rsa = min(asa / max_asa, 1.5)  # cap at 1.5 for edge cases

        ss3 = SS8_TO_SS3.get(ss, "C")

        residues.append({
            "resnum": resnum,
            "aa": aa,
            "ss8": ss,
            "ss3": ss3,
            "asa": round(asa, 1),
            "rsa": round(rsa, 4),
        })

    return residues


def run_dssp_on_all_pdbs():
    """Run mkdssp on all PDB files and collect per-residue annotations."""
    annotations_path = ANNOTATIONS_DIR / "dssp_annotations.json"

    # Check if we already have annotations
    if annotations_path.exists():
        print(f"[DSSP] Found existing annotations at {annotations_path}")
        with open(annotations_path) as f:
            annotations = json.load(f)
        print(f"[DSSP] Loaded annotations for {len(annotations)} proteins")
        return annotations

    pdb_files = sorted(STRUCTURES_DIR.glob("*.pdb"))
    print(f"[DSSP] Processing {len(pdb_files)} PDB files...")

    annotations = {}
    success = 0
    failed = 0

    for i, pdb_path in enumerate(pdb_files):
        accession = pdb_path.stem
        try:
            result = subprocess.run(
                [str(MKDSSP), "--output-format", "dssp", "--calculate-accessibility",
                 str(pdb_path)],
                capture_output=True, text=True, timeout=60,
            )

            # mkdssp writes a warning to stderr about PDB format but still works
            output = result.stdout
            if not output.strip():
                print(f"  [WARN] Empty output for {accession}")
                failed += 1
                continue

            residues = parse_dssp_classic(output)
            if not residues:
                print(f"  [WARN] No residues parsed for {accession}")
                failed += 1
                continue

            annotations[accession] = residues
            success += 1

        except subprocess.TimeoutExpired:
            print(f"  [WARN] Timeout for {accession}")
            failed += 1
        except Exception as e:
            print(f"  [WARN] Error for {accession}: {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  ... processed {i+1}/{len(pdb_files)} ({success} ok, {failed} failed)")

    print(f"[DSSP] Done: {success} succeeded, {failed} failed out of {len(pdb_files)}")

    # Save annotations
    with open(annotations_path, "w") as f:
        json.dump(annotations, f)
    print(f"[DSSP] Saved annotations to {annotations_path}")

    # Print summary statistics
    all_ss3 = []
    all_rsa = []
    for acc, res_list in annotations.items():
        for r in res_list:
            all_ss3.append(r["ss3"])
            all_rsa.append(r["rsa"])

    ss3_counts = Counter(all_ss3)
    total = len(all_ss3)
    print(f"\n[DSSP] Summary across {len(annotations)} proteins, {total} residues:")
    for ss in ["H", "E", "C"]:
        pct = 100 * ss3_counts.get(ss, 0) / total
        print(f"  SS3={ss}: {ss3_counts.get(ss, 0)} ({pct:.1f}%)")
    print(f"  Mean RSA: {np.mean(all_rsa):.3f}, Median RSA: {np.median(all_rsa):.3f}")

    return annotations


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: Load SAE models and compute structural enrichments
# ═════════════════════════════════════════════════════════════════════════════

def load_sae(sae_dir: Path):
    """Load a trained SAE model from checkpoint."""
    ckpt_path = sae_dir / "final.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # The checkpoint stores config under 'sae_config' key
    config = checkpoint["sae_config"]
    if isinstance(config, dict):
        sae = build_sae(SAEConfig(**config))
    else:
        # Already a SAEConfig object
        sae = build_sae(config)
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.eval()
    return sae


def build_annotation_lookup(dssp_annotations, residue_index):
    """Build arrays of SS and RSA aligned with activation rows.

    residue_index: list of {accession, start, end, length}
    Returns: ss3_array (str array), rsa_array (float array), valid_mask (bool array)
    """
    total_residues = residue_index[-1]["end"]
    ss3_array = np.full(total_residues, "", dtype=object)
    rsa_array = np.full(total_residues, np.nan, dtype=np.float32)
    valid_mask = np.zeros(total_residues, dtype=bool)

    for entry in residue_index:
        acc = entry["accession"]
        start = entry["start"]
        end = entry["end"]
        length = entry["length"]

        if acc not in dssp_annotations:
            continue

        dssp_res = dssp_annotations[acc]

        # Align: the activation row order corresponds to residue positions 0..length-1
        # DSSP residues are indexed by resnum (1-based in PDB)
        # We assume the activation ordering matches the sequence order
        n_dssp = len(dssp_res)
        n_act = length

        # Use min of both lengths for alignment
        n = min(n_dssp, n_act)
        for j in range(n):
            ss3_array[start + j] = dssp_res[j]["ss3"]
            rsa_array[start + j] = dssp_res[j]["rsa"]
            valid_mask[start + j] = True

    return ss3_array, rsa_array, valid_mask


def compute_feature_structural_annotations(
    sae, activations_h5_path, residue_index, dssp_annotations, top_n=200
):
    """Run SAE on activations, compute structural enrichments for top features.

    Returns dict with per-feature annotations and summary stats.
    """
    # Build annotation arrays
    ss3_array, rsa_array, valid_mask = build_annotation_lookup(
        dssp_annotations, residue_index
    )

    n_valid = valid_mask.sum()
    n_total = len(valid_mask)
    print(f"    Annotation coverage: {n_valid}/{n_total} residues "
          f"({100*n_valid/n_total:.1f}%)")

    # Load activations and run SAE
    print(f"    Loading activations from {activations_h5_path.name}...")
    with h5py.File(activations_h5_path, "r") as f:
        activations = torch.tensor(f["activations"][:], dtype=torch.float32)

    print(f"    Running SAE forward pass on {activations.shape[0]} residues...")
    batch_size = 4096
    all_z = []
    with torch.no_grad():
        for i in range(0, activations.shape[0], batch_size):
            batch = activations[i:i+batch_size]
            out = sae(batch)
            all_z.append(out["z"])

    z = torch.cat(all_z, dim=0).numpy()  # (n_residues, dict_size)
    print(f"    Feature activations shape: {z.shape}")

    # Find top-N most active features (by mean activation across all residues)
    mean_act_per_feature = z.mean(axis=0)
    top_feature_indices = np.argsort(mean_act_per_feature)[::-1][:top_n]

    # Restrict to valid residues (those with DSSP annotations)
    z_valid = z[valid_mask]
    ss3_valid = ss3_array[valid_mask]
    rsa_valid = rsa_array[valid_mask]

    # Precompute SS masks
    ss_types = ["H", "E", "C"]
    ss_masks = {}
    for ss in ss_types:
        ss_masks[ss] = (ss3_valid == ss)

    ss_counts = {ss: mask.sum() for ss, mask in ss_masks.items()}
    print(f"    SS distribution (valid): H={ss_counts['H']}, E={ss_counts['E']}, C={ss_counts['C']}")

    # Compute enrichments for each top feature
    feature_annotations = []

    for feat_idx in top_feature_indices:
        feat_idx = int(feat_idx)
        feat_acts = z_valid[:, feat_idx]
        overall_mean = feat_acts.mean()

        if overall_mean < 1e-10:
            # Dead or nearly dead feature
            feature_annotations.append({
                "feature_idx": feat_idx,
                "mean_activation": 0.0,
                "ss_enrichment": {"H": 0.0, "E": 0.0, "C": 0.0},
                "ss_preference": "C",
                "max_ss_enrichment": 0.0,
                "rsa_correlation": 0.0,
                "surface_preference": "neutral",
            })
            continue

        # SS enrichment: mean activation at SS type / overall mean
        ss_enrichment = {}
        for ss in ss_types:
            mask = ss_masks[ss]
            if mask.sum() > 0:
                ss_mean = feat_acts[mask].mean()
                ss_enrichment[ss] = round(float(ss_mean / overall_mean), 4)
            else:
                ss_enrichment[ss] = 0.0

        ss_preference = max(ss_enrichment, key=ss_enrichment.get)
        max_enrichment = ss_enrichment[ss_preference]

        # RSA correlation (Pearson)
        # Only compute where both feature and RSA are valid
        feat_for_corr = feat_acts
        rsa_for_corr = rsa_valid

        # Remove NaN RSA values
        rsa_valid_mask2 = ~np.isnan(rsa_for_corr)
        if rsa_valid_mask2.sum() > 10:
            r = np.corrcoef(feat_for_corr[rsa_valid_mask2],
                            rsa_for_corr[rsa_valid_mask2])[0, 1]
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0

        # Surface/buried preference
        if r > 0.1:
            surface_pref = "surface"
        elif r < -0.1:
            surface_pref = "buried"
        else:
            surface_pref = "neutral"

        feature_annotations.append({
            "feature_idx": feat_idx,
            "mean_activation": round(float(overall_mean), 6),
            "ss_enrichment": ss_enrichment,
            "ss_preference": ss_preference,
            "max_ss_enrichment": round(float(max_enrichment), 4),
            "rsa_correlation": round(float(r), 4),
            "surface_preference": surface_pref,
        })

    # Summary statistics
    prefs = Counter(fa["ss_preference"] for fa in feature_annotations)
    surf_prefs = Counter(fa["surface_preference"] for fa in feature_annotations)
    strong_ss = sum(1 for fa in feature_annotations if fa["max_ss_enrichment"] > 2.0)
    strong_ss_ratio = strong_ss / len(feature_annotations)

    all_rsa_corrs = [fa["rsa_correlation"] for fa in feature_annotations]

    summary = {
        "n_features_analyzed": len(feature_annotations),
        "ss_preference_counts": dict(prefs),
        "surface_preference_counts": dict(surf_prefs),
        "strong_ss_enrichment_count": strong_ss,
        "strong_ss_enrichment_fraction": round(strong_ss_ratio, 4),
        "mean_rsa_correlation": round(float(np.mean(all_rsa_corrs)), 4),
        "mean_max_ss_enrichment": round(float(np.mean([fa["max_ss_enrichment"] for fa in feature_annotations])), 4),
    }

    return {
        "features": feature_annotations,
        "summary": summary,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 & 4: Process all SAEs, save results, compare
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("DSSP-BASED STRUCTURAL ANNOTATION PIPELINE FOR ESM-3 SAE FEATURES")
    print("=" * 80)
    t0 = time.time()

    # ── Step 1: DSSP ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: Running DSSP on PDB structures")
    print("=" * 80)
    dssp_annotations = run_dssp_on_all_pdbs()

    # ── Step 2 & 3: Process each SAE ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2 & 3: SAE feature structural annotation")
    print("=" * 80)

    conditions = ["S", "S_St"]
    layers = [16, 33, 42]

    # Map condition label to activation file prefix and residue index file
    condition_to_prefix = {
        "S": "S",
        "S_St": "S+St",
    }

    all_results = {}

    for condition in conditions:
        for layer in layers:
            label = f"{condition}_layer_{layer}"
            prefix = condition_to_prefix[condition]

            print(f"\n{'─' * 60}")
            print(f"Processing: {label}")
            print(f"{'─' * 60}")

            # Paths
            sae_dir = MODELS_DIR / f"{condition}_layer_{layer}_topk"
            act_path = ACTIVATIONS_DIR / f"{prefix}_layer_{layer}.h5"
            ridx_path = ACTIVATIONS_DIR / f"{prefix}_residue_index.json"

            # Check existence
            if not sae_dir.exists():
                print(f"  [SKIP] SAE dir not found: {sae_dir}")
                continue
            if not act_path.exists():
                print(f"  [SKIP] Activation file not found: {act_path}")
                continue
            if not ridx_path.exists():
                print(f"  [SKIP] Residue index not found: {ridx_path}")
                continue

            # Load SAE
            print(f"  Loading SAE from {sae_dir.name}...")
            sae = load_sae(sae_dir)
            print(f"  SAE: input_dim={sae.config.input_dim}, "
                  f"dict_size={sae.config.dict_size}, k={sae.config.k}")

            # Load residue index
            with open(ridx_path) as f:
                residue_index = json.load(f)

            # Compute annotations
            result = compute_feature_structural_annotations(
                sae, act_path, residue_index, dssp_annotations, top_n=200
            )

            all_results[label] = result

            # Print summary
            s = result["summary"]
            print(f"\n  SUMMARY for {label}:")
            print(f"    SS preference: H={s['ss_preference_counts'].get('H', 0)}, "
                  f"E={s['ss_preference_counts'].get('E', 0)}, "
                  f"C={s['ss_preference_counts'].get('C', 0)}")
            print(f"    Strong SS enrichment (>2x): {s['strong_ss_enrichment_count']}/{s['n_features_analyzed']} "
                  f"({100*s['strong_ss_enrichment_fraction']:.1f}%)")
            print(f"    Surface preference: surface={s['surface_preference_counts'].get('surface', 0)}, "
                  f"buried={s['surface_preference_counts'].get('buried', 0)}, "
                  f"neutral={s['surface_preference_counts'].get('neutral', 0)}")
            print(f"    Mean RSA correlation: {s['mean_rsa_correlation']:.4f}")
            print(f"    Mean max SS enrichment: {s['mean_max_ss_enrichment']:.4f}")

            # Save per-SAE JSON
            out_path = RESULTS_DIR / f"{label}_structural.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to {out_path}")

    # ── Step 4: S-only vs S+St comparison ─────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 4: S-only vs S+St Comparison")
    print("=" * 80)

    comparison = {}
    for layer in layers:
        s_key = f"S_layer_{layer}"
        sst_key = f"S_St_layer_{layer}"

        if s_key not in all_results or sst_key not in all_results:
            print(f"\n  [SKIP] Missing results for layer {layer}")
            continue

        s_summary = all_results[s_key]["summary"]
        sst_summary = all_results[sst_key]["summary"]

        print(f"\n  Layer {layer}:")
        print(f"    {'Metric':<35} {'S-only':>10} {'S+St':>10} {'Delta':>10}")
        print(f"    {'─'*65}")

        # Strong SS enrichment fraction
        s_frac = s_summary["strong_ss_enrichment_fraction"]
        sst_frac = sst_summary["strong_ss_enrichment_fraction"]
        delta = sst_frac - s_frac
        print(f"    {'Strong SS enrichment (>2x) frac':<35} {s_frac:>10.3f} {sst_frac:>10.3f} {delta:>+10.3f}")

        # Mean max SS enrichment
        s_mse = s_summary["mean_max_ss_enrichment"]
        sst_mse = sst_summary["mean_max_ss_enrichment"]
        delta_mse = sst_mse - s_mse
        print(f"    {'Mean max SS enrichment':<35} {s_mse:>10.3f} {sst_mse:>10.3f} {delta_mse:>+10.3f}")

        # Mean RSA correlation magnitude
        s_rsa = abs(s_summary["mean_rsa_correlation"])
        sst_rsa = abs(sst_summary["mean_rsa_correlation"])
        delta_rsa = sst_rsa - s_rsa
        print(f"    {'Mean |RSA correlation|':<35} {s_rsa:>10.4f} {sst_rsa:>10.4f} {delta_rsa:>+10.4f}")

        # SS preference breakdown
        for ss in ["H", "E", "C"]:
            s_n = s_summary["ss_preference_counts"].get(ss, 0)
            sst_n = sst_summary["ss_preference_counts"].get(ss, 0)
            print(f"    {'SS pref = ' + ss:<35} {s_n:>10} {sst_n:>10} {sst_n - s_n:>+10}")

        # Surface/buried breakdown
        for pref in ["surface", "buried", "neutral"]:
            s_n = s_summary["surface_preference_counts"].get(pref, 0)
            sst_n = sst_summary["surface_preference_counts"].get(pref, 0)
            print(f"    {pref + ' preference':<35} {s_n:>10} {sst_n:>10} {sst_n - s_n:>+10}")

        comparison[f"layer_{layer}"] = {
            "S_strong_ss_frac": s_frac,
            "S_St_strong_ss_frac": sst_frac,
            "delta_strong_ss_frac": round(delta, 4),
            "S_mean_max_ss_enrichment": s_mse,
            "S_St_mean_max_ss_enrichment": sst_mse,
            "S_mean_abs_rsa_corr": round(s_rsa, 4),
            "S_St_mean_abs_rsa_corr": round(sst_rsa, 4),
        }

    # Overall summary
    print(f"\n{'─' * 60}")
    print("OVERALL COMPARISON:")

    s_overall_frac = []
    sst_overall_frac = []
    for layer in layers:
        s_key = f"S_layer_{layer}"
        sst_key = f"S_St_layer_{layer}"
        if s_key in all_results and sst_key in all_results:
            s_overall_frac.append(all_results[s_key]["summary"]["strong_ss_enrichment_fraction"])
            sst_overall_frac.append(all_results[sst_key]["summary"]["strong_ss_enrichment_fraction"])

    if s_overall_frac and sst_overall_frac:
        s_avg = np.mean(s_overall_frac)
        sst_avg = np.mean(sst_overall_frac)
        print(f"  Avg strong SS enrichment fraction across layers:")
        print(f"    S-only:  {s_avg:.3f}")
        print(f"    S+St:    {sst_avg:.3f}")

        if sst_avg > s_avg:
            print(f"  --> S+St SAEs learn MORE structural features (+{sst_avg-s_avg:.3f})")
        elif sst_avg < s_avg:
            print(f"  --> S-only SAEs learn MORE structural features (+{s_avg-sst_avg:.3f})")
        else:
            print(f"  --> No difference in structural feature learning")

    # Save comparison
    comparison_path = RESULTS_DIR / "s_vs_sst_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n  Saved comparison to {comparison_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
