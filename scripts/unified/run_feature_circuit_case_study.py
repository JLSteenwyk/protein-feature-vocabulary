#!/usr/bin/env python3
"""Feature circuit case study: trace 1-3 circuits through ESM-2 layers.

Selects the most interesting circuits from sparse_feature_circuits.json,
traces activation patterns across layers, and maps to specific proteins.

Usage:
    ./env/bin/python scripts/unified/run_feature_circuit_case_study.py
"""

import sys
import os
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("circuit_case")


def load_sae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def load_feature_annotations():
    """Load SAE feature annotations from phase1 if available."""
    annotations = {}
    ann_dir = ROOT / "results" / "phase1" / "modality_saes"
    for ann_file in ann_dir.glob("*.json"):
        try:
            with open(ann_file) as f:
                data = json.load(f)
            if isinstance(data, dict) and "features" in data:
                for feat in data["features"]:
                    key = f"L{data.get('layer', '?')}_f{feat.get('index', '?')}"
                    annotations[key] = feat
        except Exception:
            continue

    # Also check phase0
    ann_dir2 = ROOT / "results" / "phase0" / "sae_reproduction"
    for ann_file in ann_dir2.glob("*annotation*.json"):
        try:
            with open(ann_file) as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, feat in data.items():
                    if isinstance(feat, dict):
                        annotations[key] = feat
        except Exception:
            continue

    return annotations


def trace_circuit(sae_upstream, sae_downstream, h5_up, h5_down,
                  upstream_features, downstream_feature,
                  sequences, offsets_dict, accs, dssp_data, metadata,
                  n_proteins=50, seed=42):
    """Trace a feature circuit across layers on real proteins."""
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    rng = np.random.RandomState(seed)

    # Find proteins where downstream feature fires at a SUBSET of positions
    # (not all positions — we want non-trivial circuits)
    log.info(f"  Finding proteins where downstream f/{downstream_feature} fires selectively...")
    protein_scores = []

    with h5py.File(h5_down, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]

        for acc in accs[:500]:  # Check first 500 proteins
            start, L = offsets_dict[acc]
            if start + L > total or L < 10:
                continue
            raw = torch.tensor(acts[start:start+L], dtype=torch.float32)
            with torch.no_grad():
                z = sae_downstream.encode(raw)
            col = z[:, downstream_feature].numpy()
            max_act = float(col.max())
            n_active = int((col > 0).sum())
            frac_active = n_active / L
            # Want proteins where feature fires at 5-80% of positions (selective, not trivial)
            if n_active >= 3 and frac_active < 0.8:
                protein_scores.append((acc, max_act, n_active, frac_active))

    # Sort by max activation among selective proteins
    protein_scores.sort(key=lambda x: -x[1])
    top_proteins = [p[0] for p in protein_scores[:n_proteins]]
    log.info(f"  Found {len(protein_scores)} selective proteins (5-80% active), using top {len(top_proteins)}")

    # Trace each protein
    traces = []
    for acc in top_proteins[:10]:  # Detailed traces for top 10
        start, L = offsets_dict[acc]
        seq = sequences.get(acc, "")
        L = min(len(seq), L)
        dssp = dssp_data.get(acc, [])

        # Get downstream activations
        with h5py.File(h5_down, "r") as f:
            raw_down = torch.tensor(f["activations"][start:start+L], dtype=torch.float32)
        with torch.no_grad():
            z_down = sae_downstream.encode(raw_down).numpy()

        down_col = z_down[:, downstream_feature]
        down_active = np.where(down_col > 0)[0]

        # Get upstream activations
        with h5py.File(h5_up, "r") as f:
            total_up = f["activations"].shape[0]
            if start + L > total_up:
                continue
            raw_up = torch.tensor(f["activations"][start:start+L], dtype=torch.float32)
        with torch.no_grad():
            z_up = sae_upstream.encode(raw_up).numpy()

        upstream_traces = {}
        for uf in upstream_features:
            up_col = z_up[:, uf]
            up_active = np.where(up_col > 0)[0]
            overlap = set(down_active.tolist()) & set(up_active.tolist())
            # Compute overlap enrichment: observed / expected by chance
            expected_overlap = (len(down_active) * len(up_active)) / max(L, 1)
            enrichment = len(overlap) / max(expected_overlap, 1e-10)
            upstream_traces[int(uf)] = {
                "n_active": int(len(up_active)),
                "frac_active": float(len(up_active) / max(L, 1)),
                "active_positions": up_active.tolist()[:30],
                "overlap_with_downstream": len(overlap),
                "overlap_enrichment": float(enrichment),
                "overlap_positions": sorted(overlap)[:20],
            }

        # Annotate positions
        position_annotations = []
        for pos in sorted(set(down_active.tolist())):
            ann = {"pos": int(pos)}
            if pos < len(seq):
                ann["aa"] = seq[pos]
            if dssp and pos < len(dssp):
                ann["ss3"] = dssp[pos].get("ss3", "?")
                ann["asa"] = dssp[pos].get("asa", 0)
            ann["downstream_activation"] = float(down_col[pos])
            # Which upstream features also fire here?
            firing_upstream = []
            for uf in upstream_features:
                if z_up[pos, uf] > 0:
                    firing_upstream.append(int(uf))
            ann["firing_upstream"] = firing_upstream
            position_annotations.append(ann)

        # Metadata
        m = metadata.get(acc, {})
        pname = m.get("protein_name", acc)
        org = m.get("organism", "")

        traces.append({
            "accession": acc,
            "protein_name": pname,
            "organism": org,
            "length": L,
            "n_downstream_active": int(len(down_active)),
            "downstream_active_positions": down_active.tolist()[:50],
            "upstream_traces": upstream_traces,
            "position_annotations": position_annotations[:30],
        })

    # Aggregate statistics
    all_overlaps = []
    all_enrichments = []
    for trace in traces:
        for uf, ut in trace["upstream_traces"].items():
            if ut["n_active"] > 0 and trace["n_downstream_active"] > 0:
                overlap_frac = ut["overlap_with_downstream"] / trace["n_downstream_active"]
                all_overlaps.append(overlap_frac)
                all_enrichments.append(ut["overlap_enrichment"])

    return {
        "traces": traces,
        "n_proteins_with_downstream": len(protein_scores),
        "mean_overlap_fraction": float(np.mean(all_overlaps)) if all_overlaps else 0,
        "mean_overlap_enrichment": float(np.mean(all_enrichments)) if all_enrichments else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-circuits", type=int, default=3)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "unified" / "esm2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json") as f:
        dssp_data = json.load(f)

    accs = sorted(sequences.keys())
    offsets_dict = {}
    cur = 0
    for acc in accs:
        offsets_dict[acc] = (cur, len(sequences[acc]))
        cur += len(sequences[acc])

    # Load circuits
    with open(ROOT / "results" / "unified" / "esm2" / "sparse_feature_circuits.json") as f:
        circuits_data = json.load(f)

    lp = circuits_data["layer_pairs"]["L16_to_L24"]
    connections = lp["connections"]
    log.info(f"Found {len(connections)} downstream features with circuits")

    # Rank circuits by total causal effect, FILTERING to position-specific features
    # (TopK SAEs always activate k=64 features, so broadly-active features fire everywhere
    #  and give trivial 100% overlap — we want position-specific circuits instead)
    circuit_scores = []
    for conn in connections:
        downstream = conn["downstream_feature"]
        downstream_freq = conn.get("downstream_freq", 1.0)
        upstream_conns = conn["upstream_connections"]
        total_delta = sum(v["mean_delta"] for v in upstream_conns.values())
        n_upstream = len(upstream_conns)
        top_upstream = sorted(upstream_conns.items(), key=lambda x: -x[1]["mean_delta"])[:5]
        circuit_scores.append({
            "downstream": downstream,
            "downstream_freq": downstream_freq,
            "total_delta": total_delta,
            "n_upstream": n_upstream,
            "top_upstream": [(int(k), v["mean_delta"]) for k, v in top_upstream],
        })

    # Filter to position-specific features (fire at <50% of positions)
    specific = [c for c in circuit_scores if c["downstream_freq"] < 0.5]
    log.info(f"Position-specific circuits (freq<0.5): {len(specific)} / {len(circuit_scores)}")
    specific.sort(key=lambda x: -x["total_delta"])

    # Select top position-specific circuits
    selected = specific[:args.n_circuits]
    log.info(f"\nSelected {len(selected)} circuits:")
    for s in selected:
        log.info(f"  Downstream f/{s['downstream']}: total_delta={s['total_delta']:.1f}, "
                 f"n_upstream={s['n_upstream']}")
        for uf, delta in s["top_upstream"]:
            log.info(f"    ← f/{uf}: delta={delta:.2f}")

    # Load SAEs
    log.info("\nLoading SAEs...")
    sae_l16 = load_sae(str(ROOT / "models" / "sae" / "esm2_scaled" / "layer_16_topk" / "best.pt"))
    sae_l24 = load_sae(str(ROOT / "models" / "sae" / "esm2_scaled" / "layer_24_topk" / "best.pt"))
    h5_l16 = str(ROOT / "data" / "activations" / "esm2_scaled" / "layer_16.h5")
    h5_l24 = str(ROOT / "data" / "activations" / "esm2_scaled" / "layer_24.h5")

    # Compute L16 feature frequencies to identify position-specific features
    log.info("Computing L16 feature frequencies...")
    import h5py as h5py_temp
    with h5py_temp.File(h5_l16, "r") as f:
        sample = torch.tensor(f["activations"][:10000], dtype=torch.float32)
    with torch.no_grad():
        z_sample = sae_l16.encode(sample).numpy()
    l16_freqs = (z_sample > 0).sum(axis=0) / len(sample)

    # Re-score circuits using ONLY position-specific upstream features
    # (broadly-active upstream features fire everywhere and give trivial overlap)
    log.info("Re-scoring circuits with position-specific upstream features...")
    for circuit in circuit_scores:
        # Find the original connection data
        conn = next((c for c in connections if c["downstream_feature"] == circuit["downstream"]), None)
        if conn is None:
            circuit["specific_upstream"] = []
            continue
        specific_up = []
        for uf_key, uf_data in conn["upstream_connections"].items():
            uf = int(uf_key)
            if l16_freqs[uf] < 0.5:  # Position-specific
                specific_up.append((uf, uf_data["mean_delta"], float(l16_freqs[uf])))
        specific_up.sort(key=lambda x: -abs(x[1]))
        circuit["specific_upstream"] = specific_up[:5]
        circuit["specific_total_delta"] = sum(abs(d) for _, d, _ in specific_up)

    # Re-select: among position-specific circuits, rank by specific upstream delta
    specific = [c for c in circuit_scores if c["downstream_freq"] < 0.5 and c.get("specific_upstream")]
    specific.sort(key=lambda x: -x.get("specific_total_delta", 0))
    selected = specific[:args.n_circuits]

    log.info(f"\nSelected {len(selected)} circuits with specific upstream features:")
    for s in selected:
        log.info(f"  Downstream f/{s['downstream']} (freq={s['downstream_freq']:.3f}): "
                 f"specific_delta={s.get('specific_total_delta', 0):.1f}")
        for uf, delta, freq in s.get("specific_upstream", []):
            log.info(f"    <- f/{uf}: delta={delta:.2f}, freq={freq:.3f}")

    # Load feature annotations if available
    annotations = load_feature_annotations()

    # Trace each circuit
    case_studies = []
    for ci, circuit in enumerate(selected):
        log.info(f"\n{'='*50}")
        log.info(f"Circuit {ci+1}: downstream f/{circuit['downstream']}")
        log.info(f"{'='*50}")

        upstream_features = [uf for uf, _, _ in circuit.get("specific_upstream", circuit["top_upstream"])]

        trace_result = trace_circuit(
            sae_l16, sae_l24, h5_l16, h5_l24,
            upstream_features, circuit["downstream"],
            sequences, offsets_dict, accs, dssp_data, metadata
        )

        # Look up annotations
        down_ann = annotations.get(f"L24_f{circuit['downstream']}", {})
        up_anns = {uf: annotations.get(f"L16_f{uf}", {}) for uf in upstream_features}

        # Characterize the circuit biologically
        # What AA/SS patterns do downstream active positions have?
        aa_counter = Counter()
        ss_counter = Counter()
        for trace in trace_result["traces"]:
            for pa in trace["position_annotations"]:
                aa_counter[pa.get("aa", "?")] += 1
                ss_counter[pa.get("ss3", "?")] += 1

        case_study = {
            "circuit_index": ci,
            "downstream_feature": circuit["downstream"],
            "downstream_annotation": down_ann.get("description", "unknown") if down_ann else "unknown",
            "upstream_features": [
                {
                    "feature": uf,
                    "mean_delta": delta,
                    "annotation": up_anns.get(uf, {}).get("description", "unknown"),
                }
                for uf, delta in circuit["top_upstream"]
            ],
            "total_causal_effect": circuit["total_delta"],
            "n_proteins_with_circuit": trace_result["n_proteins_with_downstream"],
            "mean_positional_overlap": trace_result["mean_overlap_fraction"],
            "downstream_aa_distribution": dict(aa_counter.most_common(10)),
            "downstream_ss_distribution": dict(ss_counter),
            "protein_traces": trace_result["traces"],
        }
        case_studies.append(case_study)

        log.info(f"  Proteins: {trace_result['n_proteins_with_downstream']}")
        log.info(f"  Mean overlap: {trace_result['mean_overlap_fraction']:.3f}")
        log.info(f"  AA distribution: {dict(aa_counter.most_common(5))}")
        log.info(f"  SS distribution: {dict(ss_counter)}")

    # Save
    output = {
        "model": "esm2",
        "upstream_layer": 16,
        "downstream_layer": 24,
        "n_circuits": len(case_studies),
        "case_studies": case_studies,
    }

    out_path = out_dir / "feature_circuit_case_study.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


from collections import Counter

if __name__ == "__main__":
    main()
