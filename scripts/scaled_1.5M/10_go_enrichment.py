#!/usr/bin/env python3
"""GO enrichment analysis per SAE feature using gokit ORA with BH-FDR correction.

For each active SAE feature, identifies proteins where it activates strongly
and tests for GO term enrichment using over-representation analysis (ORA)
with Benjamini-Hochberg FDR correction.

Fixes over the original Fisher-exact-only version:
  - C2: Proper BH-FDR multiple testing correction via gokit OraRunner
  - M1: Same enrichment engine as 29_random_baseline.py (consistency)
  - M4: Odds ratios computed with denominator floor to avoid Inf

Usage:
    ./env/bin/python scripts/scaled_1.5M/10_go_enrichment.py
"""

import os
import sys
import json
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
from gokit.core.enrichment import OraRunner

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("go_enrichment")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

# Namespace prefix mapping from metadata term strings ("F:", "P:", "C:")
_NS_PREFIX = {"F": "MF", "P": "BP", "C": "CC"}

# FDR threshold for significance
FDR_THRESHOLD = 0.05


def build_go_mappings(protein_ids, metadata):
    """Build protein->GO and GO->namespace mappings from metadata.

    Returns:
        protein_to_go: dict[str, set[str]]  -- protein_id -> set of GO IDs
        go_to_namespace: dict[str, str]      -- GO:XXXX -> "BP"/"MF"/"CC"
    """
    protein_to_go = {}
    go_to_namespace = {}

    for pid in protein_ids:
        meta = metadata.get(pid, {})
        go_terms = set()
        for g in meta.get("go_terms", []):
            go_id = g.get("id", "")
            term_str = g.get("term", "")
            if go_id:
                go_terms.add(go_id)
                # Parse namespace from term string prefix (e.g. "F:mRNA binding")
                if go_id not in go_to_namespace and term_str:
                    prefix = term_str.split(":")[0]
                    ns = _NS_PREFIX.get(prefix, "BP")  # default BP if unknown
                    go_to_namespace[go_id] = ns
        protein_to_go[pid] = go_terms

    return protein_to_go, go_to_namespace


def safe_odds_ratio(study_count, study_n, pop_count, pop_n):
    """Compute odds ratio with denominator floor to avoid Inf (fixes M4).

    Uses enrichment ratio: (study_count/study_n) / (pop_count/pop_n)
    with a floor of 1e-6 on the denominator fraction.
    """
    study_frac = study_count / max(study_n, 1)
    pop_frac = pop_count / max(pop_n, 1)
    return study_frac / max(pop_frac, 1e-6)


def run_enrichment_for_model(model_name, metadata):
    """Run GO enrichment for all active features of one model."""
    log.info(f"\n{'='*60}")
    log.info(f"GO ENRICHMENT (gokit ORA + BH-FDR): {model_name}")
    log.info(f"{'='*60}")

    # Load protein-level feature summaries
    feat_dir = FEATURE_ROOT / model_name / "residue"
    with h5py.File(feat_dir / "protein_summaries.h5", "r") as f:
        protein_ids = [s.decode() for s in f["protein_ids"][:]]
        protein_max = f["protein_max_activations"][:]  # (n_proteins, d_sae)
        feature_counts = f["feature_activation_counts"][:]

    d_sae = protein_max.shape[1]
    n_proteins = len(protein_ids)
    active_features = np.where(feature_counts > 0)[0]
    log.info(f"  {n_proteins} proteins, {d_sae} features ({len(active_features)} active)")

    # Build GO mappings
    protein_to_go, go_to_namespace = build_go_mappings(protein_ids, metadata)

    background_proteins = set(pid for pid in protein_ids if protein_to_go.get(pid))
    log.info(f"  Proteins with GO terms: {len(background_proteins)}/{n_proteins}")
    log.info(f"  Unique GO terms: {len(go_to_namespace)}")

    if len(background_proteins) < 100:
        log.warning("  Too few proteins with GO terms, skipping")
        return None

    # Build OraRunner once (reused for all features)
    runner = OraRunner(
        population_genes=background_proteins,
        gene_to_go={pid: protein_to_go[pid] for pid in background_proteins},
        go_to_namespace=go_to_namespace,
    )

    # Background protein indices and IDs (only those with GO annotations)
    bg_idx = [i for i, pid in enumerate(protein_ids) if pid in background_proteins]
    bg_pids = [protein_ids[i] for i in bg_idx]

    enrichment_results = []
    n_enriched_features = 0

    for fi, fid in enumerate(active_features):
        # Get max activation per protein for this feature (background only)
        max_acts = protein_max[bg_idx, fid]

        # Active = top 10th percentile among non-zero activations
        nonzero_mask = max_acts > 0
        if nonzero_mask.sum() < 5:
            continue

        threshold = np.percentile(max_acts[nonzero_mask], 90)
        if threshold <= 0:
            continue

        active_pids = set(pid for pid, act in zip(bg_pids, max_acts) if act >= threshold)

        if len(active_pids) < 3:
            continue

        # Run ORA with BH-FDR correction (fixes C2)
        ora_results = runner.run_study(
            study_genes=active_pids,
            namespace_filter="all",
            method="fdr_bh",
            test_direction="over",
        )

        # Filter by adjusted p-value
        feature_enrichments = []
        for r in ora_results:
            if r.p_adjusted < FDR_THRESHOLD and r.study_count >= 2:
                feature_enrichments.append({
                    "go_id": r.go_id,
                    "pval": float(r.p_uncorrected),
                    "pval_adjusted": float(r.p_adjusted),
                    "odds_ratio": float(safe_odds_ratio(
                        r.study_count, r.study_n, r.pop_count, r.pop_n
                    )),
                    "n_overlap": r.study_count,
                    "n_active": r.study_n,
                    "n_go_term": r.pop_count,
                    "namespace": r.namespace,
                })

        if feature_enrichments:
            # Sort by adjusted p-value
            feature_enrichments.sort(key=lambda x: x["pval_adjusted"])
            n_enriched_features += 1

            enrichment_results.append({
                "feature_id": int(fid),
                "n_active_proteins": int(len(active_pids)),
                "n_enriched_terms": len(feature_enrichments),
                "top_terms": feature_enrichments[:10],
            })

        if (fi + 1) % 500 == 0:
            log.info(f"    Feature {fi+1}/{len(active_features)}: "
                     f"{n_enriched_features} features with enrichment so far")

    log.info(f"  Features with GO enrichment (FDR<{FDR_THRESHOLD}): "
             f"{n_enriched_features}/{len(active_features)} "
             f"({100*n_enriched_features/max(len(active_features),1):.1f}%)")

    return {
        "model": model_name,
        "n_proteins": n_proteins,
        "n_proteins_with_go": len(background_proteins),
        "n_active_features": len(active_features),
        "n_features_with_enrichment": n_enriched_features,
        "frac_enriched": float(n_enriched_features / max(len(active_features), 1)),
        "correction_method": "fdr_bh",
        "fdr_threshold": FDR_THRESHOLD,
        "per_feature_results": enrichment_results[:500],  # cap to keep file size reasonable
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    results = {}
    for model_name in ["esm3", "esm2"]:
        feat_path = FEATURE_ROOT / model_name / "residue" / "protein_summaries.h5"
        if not feat_path.exists():
            log.warning(f"No features for {model_name}, skipping")
            continue
        result = run_enrichment_for_model(model_name, metadata)
        if result:
            results[model_name] = result

    out_path = OUTPUT_DIR / "go_enrichment.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
