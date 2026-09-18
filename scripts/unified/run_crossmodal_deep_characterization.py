#!/usr/bin/env python3
"""Deep characterization of cross-modal vs matched SAE features.

Tests sophisticated hypotheses beyond simple structural properties:
1. pLDDT confidence (flexibility/disorder)
2. Functional site proximity (distance to nearest active/binding site)
3. Conservation proxy (MLM prediction entropy)
4. Domain boundary proximity (Pfam domain edges)
5. Residue contact network topology (degree, clustering coefficient, betweenness)

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/run_crossmodal_deep_characterization.py
"""

import sys
import os
import json
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

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
log = logging.getLogger("deep_char")
warnings.filterwarnings("ignore", category=FutureWarning)


def load_sae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def extract_plddt(pdb_path):
    """Extract per-residue pLDDT from AlphaFold PDB (stored in B-factor column)."""
    plddts = []
    seen_resnums = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26].strip())
                if resnum not in seen_resnums:
                    seen_resnums.add(resnum)
                    bfactor = float(line[60:66].strip())
                    plddts.append(bfactor)
    return np.array(plddts)


def extract_ca_coords(pdb_path):
    """Extract CA coordinates from PDB."""
    coords = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resnum = int(line[22:26].strip())
                if resnum not in seen:
                    seen.add(resnum)
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords) if coords else None


def compute_contact_graph_metrics(coords, active_positions, contact_thresh=8.0):
    """Compute network metrics for active residues in the contact graph."""
    if coords is None or len(active_positions) < 3:
        return None

    active_set = set(active_positions)
    valid_active = [p for p in active_positions if p < len(coords)]
    if len(valid_active) < 3:
        return None

    # Build subgraph adjacency for active residues
    n = len(valid_active)
    pos_to_idx = {p: i for i, p in enumerate(valid_active)}
    adj = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coords[valid_active[i]] - coords[valid_active[j]])
            if dist < contact_thresh:
                adj[i, j] = True
                adj[j, i] = True

    # Degree
    degrees = adj.sum(axis=1)
    mean_degree = degrees.mean()

    # Clustering coefficient
    clustering_coeffs = []
    for i in range(n):
        neighbors = np.where(adj[i])[0]
        k = len(neighbors)
        if k < 2:
            clustering_coeffs.append(0.0)
            continue
        # Count edges among neighbors
        edges_among = sum(adj[ni, nj] for ii, ni in enumerate(neighbors)
                         for nj in neighbors[ii + 1:])
        clustering_coeffs.append(2 * edges_among / (k * (k - 1)))
    mean_clustering = np.mean(clustering_coeffs)

    # Connected components (simple BFS)
    visited = set()
    n_components = 0
    max_component = 0
    for start in range(n):
        if start in visited:
            continue
        n_components += 1
        queue = [start]
        comp_size = 0
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            comp_size += 1
            for nb in np.where(adj[node])[0]:
                if nb not in visited:
                    queue.append(nb)
        max_component = max(max_component, comp_size)

    return {
        "mean_degree": float(mean_degree),
        "mean_clustering": float(mean_clustering),
        "n_components": int(n_components),
        "largest_component_frac": float(max_component / n),
        "n_active": n,
    }


def compute_functional_site_distances(positions, func_sites, seq_len):
    """Compute min distance from each position to nearest functional site."""
    if not func_sites:
        return None

    # Build set of functional site positions
    func_positions = set()
    for site in func_sites:
        for p in range(site["start"] - 1, site["end"]):  # 1-indexed to 0-indexed
            if 0 <= p < seq_len:
                func_positions.add(p)

    if not func_positions:
        return None

    func_arr = np.array(sorted(func_positions))
    distances = []
    for p in positions:
        if p in func_positions:
            distances.append(0)
        else:
            min_dist = np.min(np.abs(func_arr - p))
            distances.append(int(min_dist))

    return distances


def compute_domain_boundary_distances(positions, features_list, seq_len):
    """Compute min distance to nearest annotated region boundary.

    Uses UniProt feature annotations (binding sites, active sites, etc.)
    as proxy for domain-relevant boundaries.
    """
    if not features_list:
        return None

    # Collect all feature boundaries
    boundaries = set()
    for feat in features_list:
        if not isinstance(feat, dict):
            continue
        start = feat.get("start", 0)
        end = feat.get("end", 0)
        if isinstance(start, int) and isinstance(end, int):
            boundaries.add(start - 1)  # 0-indexed
            if end != start:
                boundaries.add(end - 1)

    if not boundaries:
        return None

    bound_arr = np.array(sorted(boundaries))
    distances = []
    for p in positions:
        min_dist = np.min(np.abs(bound_arr - p))
        distances.append(int(min_dist))

    return distances


def compute_mlm_entropy(model, tokenizer, sequence, device="cuda:0"):
    """Compute per-residue MLM prediction entropy as conservation proxy."""
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0]  # (L+2, vocab)

    # Remove BOS/EOS tokens
    logits = logits[1:-1]  # (L, vocab)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (L,)
    return entropy.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--n-proteins", type=int, default=200)
    parser.add_argument("--n-features", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Load cross-modal feature discovery results
    cm_path = ROOT / "results" / "unified" / "crossmodal_features" / f"crossmodal_features_L{args.layer}.json"
    with open(cm_path) as f:
        cm_data = json.load(f)

    crossmodal_idx = cm_data.get("crossmodal_feature_indices",
                                  [f["feature_idx"] for f in cm_data.get("crossmodal_features", [])])
    matched_idx = cm_data.get("matched_feature_indices",
                              [f["feature_idx"] for f in cm_data.get("matched_features_sample",
                                                                      cm_data.get("matched_features", []))])
    log.info(f"Cross-modal: {len(crossmodal_idx)}, Matched: {len(matched_idx)}")

    # Sample features
    rng = np.random.RandomState(42)
    n_cm = min(args.n_features, len(crossmodal_idx))
    n_mt = min(args.n_features, len(matched_idx))
    cm_sample = list(rng.choice(crossmodal_idx, n_cm, replace=False))
    mt_sample = list(rng.choice(matched_idx, n_mt, replace=False))

    # Load SAE
    sae_path = ROOT / "models" / "sae" / "esm3" / f"S_St_layer_{args.layer}_scaled_topk" / "best.pt"
    if not sae_path.exists():
        sae_path = ROOT / "models" / "sae" / "esm3_scaled" / f"S_St_layer_{args.layer}_topk" / "best.pt"
    log.info(f"SAE: {sae_path}")
    sae = load_sae(str(sae_path))

    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / f"S+St_layer_{args.layer}.h5")

    # Load annotations
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    dssp_path = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"
    with open(dssp_path) as f:
        dssp_data = json.load(f)
    structure_dir = ROOT / "data" / "scaled" / "structures"

    # Build offsets for H5
    accs = sorted(sequences.keys())
    offsets = {}
    pos = 0
    for acc in accs:
        L = len(sequences[acc])
        offsets[acc] = (pos, L)
        pos += L

    # Load ESM-2 for MLM entropy (conservation proxy)
    log.info("Loading ESM-2 for conservation proxy...")
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    esm2_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(args.device).eval()
    esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # ═══════════════════════════════════════════
    # Process features
    # ═══════════════════════════════════════════
    def characterize_features(feature_list, label):
        """Compute deep characteristics for a set of features."""
        log.info(f"\nCharacterizing {len(feature_list)} {label} features...")

        # Per-feature aggregated stats
        feat_stats = {fid: {
            "plddt_values": [],
            "func_distances": [],
            "domain_distances": [],
            "conservation": [],  # MLM entropy (lower = more conserved)
            "graph_metrics": [],
            "at_functional_site": 0,
            "total_residues": 0,
        } for fid in feature_list}

        with h5py.File(h5_path, "r") as h5f:
            acts_data = h5f["activations"]
            total = acts_data.shape[0]

            proteins_used = 0
            for pi, acc in enumerate(accs[:500]):
                if proteins_used >= args.n_proteins:
                    break

                start, L = offsets[acc]
                if start + L > total or L < 10:
                    continue

                seq = sequences[acc]
                meta = metadata.get(acc, {})
                dssp = dssp_data.get(acc, [])
                pdb_path = structure_dir / f"{acc}.pdb"

                # Check if we have structure
                if not pdb_path.exists():
                    continue

                # Extract pLDDT and coordinates
                plddt = extract_plddt(str(pdb_path))
                coords = extract_ca_coords(str(pdb_path))

                if plddt is None or len(plddt) == 0:
                    continue

                # Get functional sites
                func_sites = meta.get("features", [])

                # Compute MLM entropy for conservation
                if L <= 1022:  # ESM-2 max length
                    try:
                        entropy = compute_mlm_entropy(esm2_model, esm2_tokenizer, seq, args.device)
                    except Exception:
                        entropy = None
                else:
                    entropy = None

                # Encode through SAE
                raw = torch.tensor(acts_data[start:start + L], dtype=torch.float32)
                with torch.no_grad():
                    z = sae.encode(raw).numpy()

                # Process each feature
                for fid in feature_list:
                    col = z[:, fid]
                    active = np.where(col > 0)[0]
                    if len(active) < 3:
                        continue

                    fs = feat_stats[fid]
                    fs["total_residues"] += len(active)

                    # 1. pLDDT
                    valid_active = active[active < len(plddt)]
                    if len(valid_active) > 0:
                        fs["plddt_values"].extend(plddt[valid_active].tolist())

                    # 2. Functional site distances
                    func_dists = compute_functional_site_distances(active.tolist(), func_sites, L)
                    if func_dists is not None:
                        fs["func_distances"].extend(func_dists)
                        fs["at_functional_site"] += sum(1 for d in func_dists if d == 0)

                    # 3. Domain boundary distances
                    dom_dists = compute_domain_boundary_distances(active.tolist(), func_sites, L)
                    if dom_dists is not None:
                        fs["domain_distances"].extend(dom_dists)

                    # 4. Conservation (MLM entropy)
                    if entropy is not None:
                        valid_ent = active[active < len(entropy)]
                        if len(valid_ent) > 0:
                            fs["conservation"].extend(entropy[valid_ent].tolist())

                    # 5. Contact graph metrics
                    if coords is not None:
                        gm = compute_contact_graph_metrics(coords, active.tolist())
                        if gm is not None:
                            fs["graph_metrics"].append(gm)

                proteins_used += 1
                if (proteins_used) % 50 == 0:
                    log.info(f"  Processed {proteins_used}/{args.n_proteins} proteins")

        log.info(f"  Processed {proteins_used} proteins")

        # Aggregate per-feature to single values
        results = []
        for fid in feature_list:
            fs = feat_stats[fid]
            r = {"feature": fid}

            if fs["plddt_values"]:
                r["mean_plddt"] = float(np.mean(fs["plddt_values"]))
                r["std_plddt"] = float(np.std(fs["plddt_values"]))
                r["frac_low_confidence"] = float(np.mean(np.array(fs["plddt_values"]) < 70))
                r["frac_very_low_confidence"] = float(np.mean(np.array(fs["plddt_values"]) < 50))

            if fs["func_distances"]:
                r["mean_func_distance"] = float(np.mean(fs["func_distances"]))
                r["median_func_distance"] = float(np.median(fs["func_distances"]))
                r["frac_at_func_site"] = float(np.mean(np.array(fs["func_distances"]) == 0))
                r["frac_near_func_site"] = float(np.mean(np.array(fs["func_distances"]) <= 5))

            if fs["domain_distances"]:
                r["mean_domain_distance"] = float(np.mean(fs["domain_distances"]))
                r["median_domain_distance"] = float(np.median(fs["domain_distances"]))
                r["frac_at_boundary"] = float(np.mean(np.array(fs["domain_distances"]) <= 3))

            if fs["conservation"]:
                r["mean_entropy"] = float(np.mean(fs["conservation"]))
                r["std_entropy"] = float(np.std(fs["conservation"]))

            if fs["graph_metrics"]:
                r["mean_graph_degree"] = float(np.mean([g["mean_degree"] for g in fs["graph_metrics"]]))
                r["mean_graph_clustering"] = float(np.mean([g["mean_clustering"] for g in fs["graph_metrics"]]))
                r["mean_largest_component"] = float(np.mean([g["largest_component_frac"] for g in fs["graph_metrics"]]))

            if fs["total_residues"] > 0:
                r["total_residues"] = fs["total_residues"]

            results.append(r)

        return results

    cm_results = characterize_features(cm_sample, "cross-modal")
    mt_results = characterize_features(mt_sample, "matched")

    # ═══════════════════════════════════════════
    # Statistical comparison
    # ═══════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("DEEP CHARACTERIZATION: Cross-modal vs Matched")
    log.info("=" * 60)

    metrics = [
        ("mean_plddt", "pLDDT (confidence)", "higher=more confident"),
        ("frac_low_confidence", "Frac low confidence (<70)", "higher=more disordered"),
        ("frac_very_low_confidence", "Frac very low confidence (<50)", "higher=more disordered"),
        ("mean_func_distance", "Mean distance to functional site", "lower=closer to active/binding"),
        ("frac_at_func_site", "Frac at functional site", "higher=more functional"),
        ("frac_near_func_site", "Frac within 5 residues of func site", "higher=more functional"),
        ("mean_domain_distance", "Mean distance to domain boundary", "lower=at interfaces"),
        ("frac_at_boundary", "Frac at domain boundary (±3)", "higher=at interfaces"),
        ("mean_entropy", "Mean MLM entropy (conservation)", "lower=more conserved"),
        ("mean_graph_degree", "Mean contact graph degree", "higher=more connected"),
        ("mean_graph_clustering", "Mean contact clustering coeff", "higher=more clustered"),
        ("mean_largest_component", "Largest connected component frac", "higher=more contiguous"),
    ]

    comparison_results = {}
    for metric_key, label, direction in metrics:
        cm_vals = [r[metric_key] for r in cm_results if metric_key in r]
        mt_vals = [r[metric_key] for r in mt_results if metric_key in r]

        if len(cm_vals) < 10 or len(mt_vals) < 10:
            log.info(f"\n{label}: insufficient data (cm={len(cm_vals)}, mt={len(mt_vals)})")
            continue

        stat, p = stats.mannwhitneyu(cm_vals, mt_vals, alternative='two-sided')
        effect_size = (np.mean(cm_vals) - np.mean(mt_vals)) / np.sqrt(
            (np.std(cm_vals) ** 2 + np.std(mt_vals) ** 2) / 2 + 1e-10)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        log.info(f"\n{label} ({direction}):")
        log.info(f"  Cross-modal: {np.mean(cm_vals):.4f} ± {np.std(cm_vals):.4f} (n={len(cm_vals)})")
        log.info(f"  Matched:     {np.mean(mt_vals):.4f} ± {np.std(mt_vals):.4f} (n={len(mt_vals)})")
        log.info(f"  U-test p={p:.4g} {sig}  Cohen's d={effect_size:.3f}")

        comparison_results[metric_key] = {
            "label": label,
            "direction": direction,
            "crossmodal_mean": float(np.mean(cm_vals)),
            "crossmodal_std": float(np.std(cm_vals)),
            "crossmodal_n": len(cm_vals),
            "matched_mean": float(np.mean(mt_vals)),
            "matched_std": float(np.std(mt_vals)),
            "matched_n": len(mt_vals),
            "p_value": float(p),
            "cohens_d": float(effect_size),
            "significant": p < 0.05,
        }

    # Save results
    output = {
        "layer": args.layer,
        "n_crossmodal_tested": len(cm_sample),
        "n_matched_tested": len(mt_sample),
        "n_proteins": args.n_proteins,
        "comparisons": comparison_results,
        "crossmodal_features": cm_results,
        "matched_features": mt_results,
    }

    # Convert numpy/bool types for JSON serialization
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output = make_json_safe(output)
    out_path = ROOT / "results" / "unified" / "crossmodal_features" / f"deep_characterization_L{args.layer}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY OF SIGNIFICANT DIFFERENCES")
    log.info("=" * 60)
    sig_count = 0
    for key, comp in comparison_results.items():
        if comp["significant"]:
            sig_count += 1
            log.info(f"  {comp['label']}: p={comp['p_value']:.4g}, d={comp['cohens_d']:.3f}")
    if sig_count == 0:
        log.info("  No significant differences found")
    else:
        log.info(f"\n  {sig_count}/{len(comparison_results)} metrics significant (p<0.05)")


if __name__ == "__main__":
    main()
