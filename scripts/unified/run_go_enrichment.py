#!/usr/bin/env python3
"""GO enrichment analysis of SAE features: cross-modal vs matched.

Following Gujral et al. (PNAS 2025): for each SAE feature, identify
proteins where it activates, run GO ORA via GOKIT, compute monosemanticity
metrics on the GO DAG, and compare cross-modal vs matched features.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/run_go_enrichment.py
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

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
log = logging.getLogger("go_enrich")


def load_sae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def build_go_mappings(metadata):
    """Build gene_to_go and go_to_namespace from our metadata."""
    gene_to_go = {}
    go_to_namespace = {}

    ns_map = {"C": "cellular_component", "F": "molecular_function", "P": "biological_process"}

    for acc, meta in metadata.items():
        go_ids = set()
        for g in meta.get("go_terms", []):
            go_id = g["id"]
            go_ids.add(go_id)
            # Parse namespace from term prefix
            term = g.get("term", "")
            if ":" in term:
                prefix = term.split(":")[0]
                if prefix in ns_map:
                    go_to_namespace[go_id] = ns_map[prefix]
        if go_ids:
            gene_to_go[acc] = go_ids

    return gene_to_go, go_to_namespace


def compute_monosemanticity(enriched_go_ids, go_graph):
    """Compute monosemanticity metrics using GO DAG shortest paths.

    Lower mean shortest path = more monosemantic (terms are close in DAG).
    """
    if len(enriched_go_ids) < 2:
        return {"mean_shortest_path": None, "n_enriched": len(enriched_go_ids)}

    import networkx as nx

    # Filter to GO IDs that exist in graph
    valid_ids = [g for g in enriched_go_ids if g in go_graph]
    if len(valid_ids) < 2:
        return {"mean_shortest_path": None, "n_enriched": len(enriched_go_ids)}

    # Compute pairwise shortest path lengths (undirected)
    ug = go_graph.to_undirected()
    path_lengths = []
    for i in range(len(valid_ids)):
        for j in range(i + 1, len(valid_ids)):
            try:
                sp = nx.shortest_path_length(ug, valid_ids[i], valid_ids[j])
                path_lengths.append(sp)
            except nx.NetworkXNoPath:
                pass

    if not path_lengths:
        return {"mean_shortest_path": None, "n_enriched": len(enriched_go_ids)}

    return {
        "mean_shortest_path": float(np.mean(path_lengths)),
        "median_shortest_path": float(np.median(path_lengths)),
        "n_enriched": len(enriched_go_ids),
        "n_pairs_computed": len(path_lengths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-features", type=int, default=200,
                        help="Total features to analyze (100 per group)")
    parser.add_argument("--n-proteins", type=int, default=0,
                        help="Number of proteins to scan (0=all)")
    parser.add_argument("--activation-threshold-pct", type=float, default=90,
                        help="Percentile threshold for 'active' protein")
    parser.add_argument("--max-active-frac", type=float, default=0.5,
                        help="Skip features where >this fraction of proteins are active (too broad)")
    args = parser.parse_args()

    from gokit.core.enrichment import run_ora
    from gokit.io.obo import read_obo_graph, read_obo_namespace_map

    # Load cross-modal/matched indices
    cm_path = ROOT / "results" / "unified" / "crossmodal_features" / "crossmodal_features_L33.json"
    with open(cm_path) as f:
        cm_data = json.load(f)
    crossmodal_idx = cm_data["crossmodal_feature_indices"]
    matched_idx = cm_data["matched_feature_indices"]
    log.info(f"Cross-modal: {len(crossmodal_idx)}, Matched: {len(matched_idx)}")

    # Sample features
    rng = np.random.RandomState(42)
    n_per = min(args.n_features // 2, len(crossmodal_idx), len(matched_idx))
    cm_sample = list(rng.choice(crossmodal_idx, n_per, replace=False))
    mt_sample = list(rng.choice(matched_idx, n_per, replace=False))
    all_features = cm_sample + mt_sample
    cm_set = set(cm_sample)
    log.info(f"Sampled {n_per} cross-modal + {n_per} matched features")

    # Load data
    sae_path = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_scaled_topk" / "best.pt"
    sae = load_sae(str(sae_path))
    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33.h5")

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    all_accs = sorted(sequences.keys())
    if args.n_proteins > 0:
        all_accs = all_accs[:args.n_proteins]
    offsets = {}
    pos = 0
    for acc in sorted(sequences.keys()):
        L = len(sequences[acc])
        offsets[acc] = (pos, L)
        pos += L

    # Build GO mappings
    gene_to_go, go_to_namespace = build_go_mappings(metadata)
    log.info(f"Proteins with GO annotations: {len(gene_to_go)}")
    log.info(f"Unique GO terms: {len(go_to_namespace)}")

    # Also read namespace map from OBO for completeness
    obo_path = ROOT / "data" / "go" / "go-basic.obo"
    obo_result = read_obo_namespace_map(obo_path)
    # read_obo_namespace_map returns (dict, OboMeta) tuple
    obo_ns_map = obo_result[0] if isinstance(obo_result, tuple) else obo_result
    # Merge: prefer OBO namespace if available
    for go_id, ns in obo_ns_map.items():
        go_to_namespace[go_id] = ns
    log.info(f"GO terms with namespace (after OBO merge): {len(go_to_namespace)}")

    # Build GO DAG for monosemanticity
    log.info("Building GO DAG from OBO...")
    import networkx as nx
    obo_graph_result = read_obo_graph(obo_path)
    # Returns (namespace_map, parents_map, OboMeta)
    _, parents_map, _ = obo_graph_result
    go_graph = nx.DiGraph()
    for child, parents in parents_map.items():
        go_graph.add_node(child)
        for parent in parents:
            go_graph.add_edge(child, parent)
    log.info(f"GO DAG: {go_graph.number_of_nodes()} nodes, {go_graph.number_of_edges()} edges")

    # ═══════════════════════════════════════════════════
    # Single-pass: compute per-feature max activations per protein
    # ═══════════════════════════════════════════════════
    log.info(f"\nScanning {len(all_accs)} proteins for {len(all_features)} features...")

    # Per-feature: {fid: {acc: max_activation}}
    feature_protein_acts = {fid: {} for fid in all_features}

    with h5py.File(h5_path, "r") as f:
        acts_data = f["activations"]
        total_residues = acts_data.shape[0]

        for pi, acc in enumerate(all_accs):
            start, L = offsets[acc]
            if start + L > total_residues or L < 5:
                continue

            raw = torch.tensor(acts_data[start:start + L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()

            for fid in all_features:
                max_act = float(z[:, fid].max())
                feature_protein_acts[fid][acc] = max_act

            if (pi + 1) % 100 == 0:
                log.info(f"  Scanned {pi + 1}/{len(all_accs)} proteins")

    log.info("  Scan complete.")

    # ═══════════════════════════════════════════════════
    # Determine activation thresholds and active protein sets
    # ═══════════════════════════════════════════════════
    population_genes = set(acc for acc in all_accs if acc in gene_to_go)
    log.info(f"Population (proteins with GO terms): {len(population_genes)}")

    # ═══════════════════════════════════════════════════
    # Run GO enrichment per feature
    # ═══════════════════════════════════════════════════
    log.info("\nRunning GO enrichment per feature...")

    results = []
    ns_names = ["biological_process", "molecular_function", "cellular_component"]

    for fi, fid in enumerate(all_features):
        acts = feature_protein_acts[fid]
        if not acts:
            continue

        # Threshold: top proteins by max activation
        act_values = np.array(list(acts.values()))
        threshold = np.percentile(act_values, args.activation_threshold_pct)

        # Active protein set (must have GO annotations)
        study_genes = set(
            acc for acc, val in acts.items()
            if val > threshold and acc in gene_to_go
        )

        # Skip broadly-active features (too generic to be informative)
        active_frac = len(study_genes) / max(len(population_genes), 1)
        if active_frac > args.max_active_frac:
            results.append({
                "feature_id": fid,
                "label": "cross-modal" if fid in cm_set else "matched",
                "n_active_proteins": len(study_genes),
                "skipped": "too_broadly_active",
                "active_fraction": float(active_frac),
                "enriched_terms": [],
                "n_enriched_total": 0,
                "n_enriched_bp": 0, "n_enriched_mf": 0, "n_enriched_cc": 0,
                "monosemanticity": {"mean_shortest_path": None, "n_enriched": 0},
            })
            continue

        if len(study_genes) < 3:
            results.append({
                "feature_id": fid,
                "label": "cross-modal" if fid in cm_set else "matched",
                "n_active_proteins": len(study_genes),
                "enriched_terms": [],
                "n_enriched_bp": 0, "n_enriched_mf": 0, "n_enriched_cc": 0,
                "monosemanticity": {"mean_shortest_path": None, "n_enriched": 0},
            })
            continue

        # Run ORA for each namespace
        all_enriched = []
        ns_counts = Counter()

        for ns in ns_names:
            try:
                enrichment = run_ora(
                    study_genes=study_genes,
                    population_genes=population_genes,
                    gene_to_go=gene_to_go,
                    go_to_namespace=go_to_namespace,
                    namespace_filter=ns,
                    method="fdr_bh",
                    test_direction="over",
                )
                sig = [e for e in enrichment if e.p_adjusted < 0.05]
                for e in sig:
                    all_enriched.append({
                        "go_id": e.go_id,
                        "namespace": e.namespace,
                        "study_count": e.study_count,
                        "study_n": e.study_n,
                        "pop_count": e.pop_count,
                        "pop_n": e.pop_n,
                        "p_adjusted": float(e.p_adjusted),
                    })
                    ns_counts[ns] += 1
            except Exception as ex:
                log.warning(f"  ORA error for f/{fid} ns={ns}: {ex}")

        # Monosemanticity
        enriched_go_ids = [e["go_id"] for e in all_enriched]
        mono = compute_monosemanticity(enriched_go_ids, go_graph)

        results.append({
            "feature_id": fid,
            "label": "cross-modal" if fid in cm_set else "matched",
            "n_active_proteins": len(study_genes),
            "enriched_terms": all_enriched,
            "n_enriched_total": len(all_enriched),
            "n_enriched_bp": ns_counts.get("biological_process", 0),
            "n_enriched_mf": ns_counts.get("molecular_function", 0),
            "n_enriched_cc": ns_counts.get("cellular_component", 0),
            "monosemanticity": mono,
        })

        if (fi + 1) % 20 == 0:
            n_with_go = sum(1 for r in results if r.get("n_enriched_total", 0) > 0)
            log.info(f"  Processed {fi + 1}/{len(all_features)} features | "
                     f"{n_with_go} have GO enrichment")

    # ═══════════════════════════════════════════════════
    # Compare cross-modal vs matched
    # ═══════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("GO ENRICHMENT COMPARISON")
    log.info("=" * 60)

    cm_results = [r for r in results if r["label"] == "cross-modal"]
    mt_results = [r for r in results if r["label"] == "matched"]

    # Number of enriched terms
    cm_n_enriched = [r.get("n_enriched_total", 0) for r in cm_results]
    mt_n_enriched = [r.get("n_enriched_total", 0) for r in mt_results]

    log.info(f"\nCross-modal: {len(cm_results)} features")
    log.info(f"  Features with any GO enrichment: {sum(1 for n in cm_n_enriched if n > 0)} "
             f"({100*sum(1 for n in cm_n_enriched if n > 0)/max(len(cm_n_enriched),1):.1f}%)")
    log.info(f"  Mean enriched terms: {np.mean(cm_n_enriched):.1f} +/- {np.std(cm_n_enriched):.1f}")

    log.info(f"\nMatched: {len(mt_results)} features")
    log.info(f"  Features with any GO enrichment: {sum(1 for n in mt_n_enriched if n > 0)} "
             f"({100*sum(1 for n in mt_n_enriched if n > 0)/max(len(mt_n_enriched),1):.1f}%)")
    log.info(f"  Mean enriched terms: {np.mean(mt_n_enriched):.1f} +/- {np.std(mt_n_enriched):.1f}")

    stat, pval = stats.mannwhitneyu(cm_n_enriched, mt_n_enriched, alternative="two-sided")
    log.info(f"  Mann-Whitney (n_enriched): U={stat:.0f}, p={pval:.4f}")

    # Namespace distribution
    cm_bp = sum(r.get("n_enriched_bp", 0) for r in cm_results)
    cm_mf = sum(r.get("n_enriched_mf", 0) for r in cm_results)
    cm_cc = sum(r.get("n_enriched_cc", 0) for r in cm_results)
    mt_bp = sum(r.get("n_enriched_bp", 0) for r in mt_results)
    mt_mf = sum(r.get("n_enriched_mf", 0) for r in mt_results)
    mt_cc = sum(r.get("n_enriched_cc", 0) for r in mt_results)

    log.info(f"\nNamespace distribution:")
    log.info(f"  Cross-modal: BP={cm_bp}, MF={cm_mf}, CC={cm_cc}")
    log.info(f"  Matched:     BP={mt_bp}, MF={mt_mf}, CC={mt_cc}")

    # Chi-square on namespace distribution
    if (cm_bp + cm_mf + cm_cc > 0) and (mt_bp + mt_mf + mt_cc > 0):
        chi2, chi_p = stats.chisquare(
            [cm_bp + 1, cm_mf + 1, cm_cc + 1],
            f_exp=[(cm_bp + mt_bp) / 2 + 1, (cm_mf + mt_mf) / 2 + 1, (cm_cc + mt_cc) / 2 + 1]
        )
        log.info(f"  Chi-square namespace: chi2={chi2:.2f}, p={chi_p:.4f}")

    # Monosemanticity comparison
    cm_mono = [r["monosemanticity"]["mean_shortest_path"] for r in cm_results
               if r["monosemanticity"]["mean_shortest_path"] is not None]
    mt_mono = [r["monosemanticity"]["mean_shortest_path"] for r in mt_results
               if r["monosemanticity"]["mean_shortest_path"] is not None]

    if cm_mono and mt_mono:
        log.info(f"\nMonosemanticity (mean shortest path in GO DAG):")
        log.info(f"  Cross-modal: {np.mean(cm_mono):.2f} +/- {np.std(cm_mono):.2f} (n={len(cm_mono)})")
        log.info(f"  Matched:     {np.mean(mt_mono):.2f} +/- {np.std(mt_mono):.2f} (n={len(mt_mono)})")
        stat, pval_mono = stats.mannwhitneyu(cm_mono, mt_mono, alternative="two-sided")
        log.info(f"  Mann-Whitney: U={stat:.0f}, p={pval_mono:.4f}")
        log.info(f"  (Lower = more monosemantic/focused)")

    # Unique GO terms per group
    cm_go_ids = set()
    mt_go_ids = set()
    for r in cm_results:
        for e in r.get("enriched_terms", []):
            cm_go_ids.add(e["go_id"])
    for r in mt_results:
        for e in r.get("enriched_terms", []):
            mt_go_ids.add(e["go_id"])

    cm_only = cm_go_ids - mt_go_ids
    mt_only = mt_go_ids - cm_go_ids
    shared = cm_go_ids & mt_go_ids

    log.info(f"\nUnique GO terms:")
    log.info(f"  Cross-modal only: {len(cm_only)}")
    log.info(f"  Matched only:     {len(mt_only)}")
    log.info(f"  Shared:           {len(shared)}")

    # Top GO terms per group (by frequency across features)
    cm_go_freq = Counter()
    mt_go_freq = Counter()
    for r in cm_results:
        for e in r.get("enriched_terms", []):
            cm_go_freq[e["go_id"]] += 1
    for r in mt_results:
        for e in r.get("enriched_terms", []):
            mt_go_freq[e["go_id"]] += 1

    log.info(f"\nTop GO terms (cross-modal, by feature count):")
    for go_id, count in cm_go_freq.most_common(10):
        ns = go_to_namespace.get(go_id, "?")
        log.info(f"  {go_id} ({ns}): {count} features")

    log.info(f"\nTop GO terms (matched, by feature count):")
    for go_id, count in mt_go_freq.most_common(10):
        ns = go_to_namespace.get(go_id, "?")
        log.info(f"  {go_id} ({ns}): {count} features")

    # ═══════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════
    output = {
        "n_crossmodal": len(cm_results),
        "n_matched": len(mt_results),
        "n_proteins": len(all_accs),
        "activation_threshold_percentile": args.activation_threshold_pct,
        "comparison": {
            "cm_mean_enriched": float(np.mean(cm_n_enriched)),
            "mt_mean_enriched": float(np.mean(mt_n_enriched)),
            "enrichment_count_mannwhitney_p": float(pval),
            "cm_namespace": {"BP": cm_bp, "MF": cm_mf, "CC": cm_cc},
            "mt_namespace": {"BP": mt_bp, "MF": mt_mf, "CC": mt_cc},
            "cm_monosemanticity_mean": float(np.mean(cm_mono)) if cm_mono else None,
            "mt_monosemanticity_mean": float(np.mean(mt_mono)) if mt_mono else None,
            "monosemanticity_mannwhitney_p": float(pval_mono) if cm_mono and mt_mono else None,
            "cm_unique_go_terms": len(cm_only),
            "mt_unique_go_terms": len(mt_only),
            "shared_go_terms": len(shared),
        },
        "per_feature_results": results,
        "top_cm_go_terms": [{"go_id": g, "count": c} for g, c in cm_go_freq.most_common(20)],
        "top_mt_go_terms": [{"go_id": g, "count": c} for g, c in mt_go_freq.most_common(20)],
    }

    def make_safe(obj):
        if isinstance(obj, dict):
            return {str(k): make_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_safe(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    out_path = ROOT / "results" / "unified" / "crossmodal_features" / "go_enrichment_L33.json"
    with open(out_path, "w") as f:
        json.dump(make_safe(output), f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
