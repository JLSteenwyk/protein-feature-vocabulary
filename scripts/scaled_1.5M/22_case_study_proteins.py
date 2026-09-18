#!/usr/bin/env python3
"""Case study proteins: detailed SAE feature analysis on well-studied proteins.

Picks 5 well-studied proteins, shows per-residue SAE feature activations,
highlights structure-enhanced features, and generates per-protein summaries.

Usage:
    ./env/bin/python scripts/scaled_1.5M/22_case_study_proteins.py
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
from scipy import sparse

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("case_study")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

# Well-studied proteins to look for in our eval set
# These are common Swiss-Prot entries with extensive annotation
TARGET_PROTEINS = {
    "lysozyme": ["P00698", "P61626", "P00720"],  # hen, human, T4
    "kinase": ["P00519", "P06239", "P04629"],     # ABL1, LCK, NTRK1
    "GPCR": ["P07550", "P08172", "P25021"],       # ADRB2, CHRM2, PTH1R
    "hemoglobin": ["P69905", "P68871", "P01942"],  # HBA, HBB, MYG
    "protease": ["P00760", "P07477", "P00766"],    # trypsin, TRY1, chymotrypsin
    "GFP": ["P42212"],                              # avGFP
    "ubiquitin": ["P0CG48", "P62988"],
    "thioredoxin": ["P10599", "P0AA25"],
    "ferritin": ["P02794", "P09528"],
    "superoxide_dismutase": ["P00441", "P04179"],
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    # Load cross-modal features
    cm_path = OUTPUT_DIR / "cross_modal_features.json"
    enhanced_set = set()
    suppressed_set = set()
    if cm_path.exists():
        with open(cm_path) as f:
            cm_data = json.load(f)
        enhanced_set = set(cm_data.get("enhanced_feature_ids", []))
        suppressed_set = set(cm_data.get("suppressed_feature_ids", []))

    # Load probe weights
    probe_path = OUTPUT_DIR / "interpretable_probing.json"
    probe_weights = {}
    if probe_path.exists():
        with open(probe_path) as f:
            probe_data = json.load(f)
        if "esm3" in probe_data:
            for feat in probe_data["esm3"].get("top_features", []):
                probe_weights[feat["feature_id"]] = feat["weight"]

    # Find which target proteins are in our eval set
    eval_accs = set(sequences.keys())
    found_proteins = {}
    for category, accessions in TARGET_PROTEINS.items():
        for acc in accessions:
            if acc in eval_accs:
                found_proteins[acc] = {
                    "category": category,
                    "accession": acc,
                    "name": metadata.get(acc, {}).get("protein_name", ""),
                    "organism": metadata.get(acc, {}).get("organism", ""),
                    "length": len(sequences[acc]),
                }

    log.info(f"Found {len(found_proteins)} target proteins in eval set:")
    for acc, info in found_proteins.items():
        log.info(f"  {acc}: {info['name'][:60]} ({info['category']})")

    if not found_proteins:
        # Fallback: pick well-annotated proteins by feature count
        log.info("No target proteins found, selecting by annotation density...")
        annotated = [(acc, len(m.get("features", [])), m.get("protein_name", ""))
                     for acc, m in metadata.items() if m.get("features")]
        annotated.sort(key=lambda x: -x[1])
        for acc, n_feat, name in annotated[:5]:
            found_proteins[acc] = {
                "category": "highly_annotated",
                "accession": acc,
                "name": name,
                "organism": metadata[acc].get("organism", ""),
                "length": len(sequences[acc]),
            }

    # Load sparse SAE features
    feat_dir = FEATURE_ROOT / "esm3" / "residue"
    X_sparse = sparse.load_npz(str(feat_dir / "features_sparse.npz"))
    with h5py.File(feat_dir / "protein_summaries.h5", "r") as f:
        protein_ids = [s.decode() for s in f["protein_ids"][:]]
        offsets = f["offsets"][:]

    pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    # Analyze each case study protein
    case_studies = []

    for acc, info in found_proteins.items():
        if acc not in pid_to_idx:
            continue

        pi = pid_to_idx[acc]
        start, L = offsets[pi]

        # Get per-residue features for this protein
        X_protein = X_sparse[start:start + L].toarray()  # (L, d_sae)

        # Top features by max activation across residues
        max_per_feature = X_protein.max(axis=0)
        top_feature_idx = np.argsort(-max_per_feature)
        top_features = []

        for rank, fid in enumerate(top_feature_idx[:30]):
            if max_per_feature[fid] == 0:
                break

            # Per-residue activation for this feature
            activations = X_protein[:, fid]
            active_positions = np.where(activations > 0)[0]

            # Classify feature
            is_enhanced = int(fid) in enhanced_set
            is_suppressed = int(fid) in suppressed_set
            label = "structure-enhanced" if is_enhanced else ("structure-suppressed" if is_suppressed else "invariant")

            top_features.append({
                "rank": rank + 1,
                "feature_id": int(fid),
                "max_activation": float(max_per_feature[fid]),
                "mean_activation": float(activations[activations > 0].mean()) if activations.any() else 0,
                "n_active_residues": int(len(active_positions)),
                "active_positions": active_positions.tolist()[:20],
                "cross_modal_label": label,
                "probe_weight": probe_weights.get(int(fid), 0.0),
            })

        # Functional site overlay
        func_sites = {}
        for feat in metadata.get(acc, {}).get("features", []):
            feat_type = feat["type"]
            for pos in range(feat["start"] - 1, feat["end"]):
                if 0 <= pos < L:
                    func_sites.setdefault(pos, []).append(feat_type)

        # Which features are active at functional sites?
        func_site_features = defaultdict(list)
        for pos, types in func_sites.items():
            active_feats = np.where(X_protein[pos] > 0)[0]
            for fid in active_feats:
                for t in types:
                    func_site_features[int(fid)].append(t)

        # Summary
        n_enhanced_active = sum(1 for f in top_features if f["cross_modal_label"] == "structure-enhanced")

        case_study = {
            **info,
            "n_total_active_features": int((max_per_feature > 0).sum()),
            "n_enhanced_active": n_enhanced_active,
            "n_functional_sites": len(func_sites),
            "functional_site_types": dict(defaultdict(int, {t: 0 for _ in func_sites.values() for t in _})),
            "top_features": top_features,
            "sequence": sequences[acc],
        }

        case_studies.append(case_study)

        log.info(f"\n  {acc} ({info['name'][:50]}):")
        log.info(f"    Length: {L}, Active features: {(max_per_feature > 0).sum()}")
        log.info(f"    Functional sites: {len(func_sites)}")
        log.info(f"    Top 5 features: " +
                 ", ".join(f"f/{f['feature_id']}({f['cross_modal_label'][:3]})"
                           for f in top_features[:5]))

    output = {
        "n_case_studies": len(case_studies),
        "case_studies": case_studies,
    }

    out_path = OUTPUT_DIR / "case_study_proteins.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
