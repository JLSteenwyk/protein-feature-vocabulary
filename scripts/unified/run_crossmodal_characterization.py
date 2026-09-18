#!/usr/bin/env python3
"""Deeper structural characterization of cross-modal SAE features.

Tests specific structural hypotheses: 3D contact enrichment, burial,
loop/turn detection, spatial clustering. Compares cross-modal vs matched features.

Usage:
    ./env/bin/python scripts/unified/run_crossmodal_characterization.py --layer 33
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
from scipy import stats
from Bio.PDB import PDBParser

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crossmodal_char")


def load_sae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def parse_ca_coords(pdb_path):
    """Extract CA coordinates from PDB file."""
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("p", str(pdb_path))
    except Exception:
        return None
    coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_vector().get_array())
        break
    return np.array(coords) if coords else None


def compute_contact_map(ca_coords, threshold=8.0):
    """Compute binary contact map from CA coordinates."""
    if ca_coords is None or len(ca_coords) < 2:
        return None
    from scipy.spatial.distance import pdist, squareform
    dist = squareform(pdist(ca_coords))
    # Exclude nearby residues (|i-j| <= 5)
    n = len(ca_coords)
    mask = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 5
    contacts = (dist < threshold) & mask
    return contacts


def characterize_structural_features(sae, h5_path, feature_indices, sequences,
                                      offsets_dict, accs, dssp_data, metadata,
                                      structure_dir, n_proteins=200, seed=42):
    """Compute structural metrics for a set of features."""
    rng = np.random.RandomState(seed)
    # Select proteins with structures
    proteins_with_structure = [acc for acc in accs
                               if (structure_dir / f"{acc}.pdb").exists()]
    sample = list(rng.choice(proteins_with_structure,
                             min(n_proteins, len(proteins_with_structure)), replace=False))

    n_feats = len(feature_indices)
    fi_to_local = {fi: i for i, fi in enumerate(feature_indices)}

    # Per-feature accumulators
    contact_enrichments = [[] for _ in range(n_feats)]  # ratio of contacts in active pairs
    spatial_clusterings = [[] for _ in range(n_feats)]  # mean pairwise dist of active vs random
    ss_transition_fracs = [[] for _ in range(n_feats)]  # fraction at SS transitions
    burial_fracs = [[] for _ in range(n_feats)]  # fraction buried (ASA<20)
    surface_fracs = [[] for _ in range(n_feats)]  # fraction surface (ASA>40)
    loop_fracs = [[] for _ in range(n_feats)]  # fraction in coil
    beta_turn_fracs = [[] for _ in range(n_feats)]  # fraction at helix-strand boundaries

    with h5py.File(h5_path, "r") as f:
        acts_ds = f["activations"]
        total = acts_ds.shape[0]

        for pi, acc in enumerate(sample):
            start, L = offsets_dict[acc]
            seq = sequences[acc]
            L = min(len(seq), L)
            if start + L > total:
                continue

            # Load structure
            ca_coords = parse_ca_coords(structure_dir / f"{acc}.pdb")
            contacts = compute_contact_map(ca_coords) if ca_coords is not None else None

            # DSSP
            dssp = dssp_data.get(acc, [])

            # Encode
            raw = torch.tensor(acts_ds[start:start+L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()

            for fi in feature_indices:
                li = fi_to_local[fi]
                col = z[:, fi]
                active_pos = np.where(col > 0)[0]
                n_act = len(active_pos)
                if n_act < 3:
                    continue

                # 1. Contact enrichment
                if contacts is not None and len(ca_coords) >= L:
                    # Fraction of active-active pairs in contact
                    active_in_range = active_pos[active_pos < len(ca_coords)]
                    if len(active_in_range) >= 2:
                        n_pairs = 0
                        n_contacts = 0
                        for i in range(len(active_in_range)):
                            for j in range(i + 1, len(active_in_range)):
                                pi2, pj = active_in_range[i], active_in_range[j]
                                if abs(pi2 - pj) > 5 and pi2 < contacts.shape[0] and pj < contacts.shape[1]:
                                    n_pairs += 1
                                    if contacts[pi2, pj]:
                                        n_contacts += 1
                        if n_pairs > 0:
                            # Background: fraction of all long-range pairs in contact
                            total_contacts = contacts.sum() / 2
                            total_pairs = (contacts.shape[0] * (contacts.shape[0] - 1) / 2 -
                                          sum(max(0, contacts.shape[0] - k) for k in range(1, 6)))
                            bg_rate = total_contacts / max(total_pairs, 1)
                            obs_rate = n_contacts / n_pairs
                            if bg_rate > 0:
                                contact_enrichments[li].append(obs_rate / bg_rate)

                # 2. Spatial clustering
                if ca_coords is not None and len(ca_coords) >= L:
                    active_in_range = active_pos[active_pos < len(ca_coords)]
                    if len(active_in_range) >= 3:
                        # Mean pairwise distance of active residues
                        active_coords = ca_coords[active_in_range]
                        from scipy.spatial.distance import pdist
                        active_dists = pdist(active_coords)
                        mean_active_dist = float(active_dists.mean())

                        # Random baseline (same number of residues)
                        all_positions = np.arange(min(L, len(ca_coords)))
                        rand_dists = []
                        for _ in range(min(50, len(all_positions))):
                            rand_pos = rng.choice(all_positions, len(active_in_range), replace=False)
                            rand_coords = ca_coords[rand_pos]
                            rand_dists.append(float(pdist(rand_coords).mean()))
                        if rand_dists:
                            mean_rand_dist = np.mean(rand_dists)
                            clustering_ratio = mean_active_dist / (mean_rand_dist + 1e-10)
                            spatial_clusterings[li].append(clustering_ratio)

                # 3. SS transition enrichment
                if dssp and len(dssp) >= L:
                    n_transitions = 0
                    for pos in active_pos:
                        if pos < len(dssp) and pos > 0:
                            curr_ss = dssp[pos].get("ss3", "C")
                            prev_ss = dssp[pos-1].get("ss3", "C") if pos-1 < len(dssp) else "C"
                            next_ss = dssp[pos+1].get("ss3", "C") if pos+1 < len(dssp) else "C"
                            if curr_ss != prev_ss or curr_ss != next_ss:
                                n_transitions += 1
                    ss_transition_fracs[li].append(n_transitions / n_act)

                # 4. Burial/surface
                if dssp and len(dssp) >= L:
                    n_buried = 0
                    n_surface = 0
                    n_coil = 0
                    for pos in active_pos:
                        if pos < len(dssp):
                            asa = dssp[pos].get("asa", 0)
                            if asa is not None:
                                if asa < 20:
                                    n_buried += 1
                                elif asa > 40:
                                    n_surface += 1
                            ss3 = dssp[pos].get("ss3", "C")
                            if ss3 == "C":
                                n_coil += 1
                    burial_fracs[li].append(n_buried / n_act)
                    surface_fracs[li].append(n_surface / n_act)
                    loop_fracs[li].append(n_coil / n_act)

            if (pi + 1) % 50 == 0:
                log.info(f"  Processed {pi+1}/{len(sample)} proteins")

    # Compile per-feature results
    results = []
    for i, fi in enumerate(feature_indices):
        entry = {"feature_idx": int(fi)}

        if contact_enrichments[i]:
            entry["mean_contact_enrichment"] = float(np.mean(contact_enrichments[i]))
            entry["n_contact_observations"] = len(contact_enrichments[i])
        else:
            entry["mean_contact_enrichment"] = None
            entry["n_contact_observations"] = 0

        if spatial_clusterings[i]:
            entry["mean_spatial_clustering"] = float(np.mean(spatial_clusterings[i]))
            entry["n_clustering_observations"] = len(spatial_clusterings[i])
        else:
            entry["mean_spatial_clustering"] = None
            entry["n_clustering_observations"] = 0

        if ss_transition_fracs[i]:
            entry["mean_ss_transition_frac"] = float(np.mean(ss_transition_fracs[i]))
        else:
            entry["mean_ss_transition_frac"] = None

        if burial_fracs[i]:
            entry["mean_burial_frac"] = float(np.mean(burial_fracs[i]))
            entry["mean_surface_frac"] = float(np.mean(surface_fracs[i]))
            entry["mean_loop_frac"] = float(np.mean(loop_fracs[i]))
        else:
            entry["mean_burial_frac"] = None
            entry["mean_surface_frac"] = None
            entry["mean_loop_frac"] = None

        results.append(entry)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--n-features", type=int, default=300,
                        help="Max features to characterize per group")
    parser.add_argument("--n-proteins", type=int, default=200)
    args = parser.parse_args()

    out_dir = ROOT / "results" / "unified" / "crossmodal_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = ROOT / "data" / "scaled" / "structures"

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

    # Load cross-modal results
    cm_path = out_dir / f"crossmodal_features_L{args.layer}.json"
    with open(cm_path) as f:
        cm_data = json.load(f)

    crossmodal_idx = cm_data["crossmodal_feature_indices"]
    matched_idx = cm_data.get("matched_feature_indices", [])
    log.info(f"Cross-modal features: {len(crossmodal_idx)}, Matched: {len(matched_idx)}")

    # Subsample cross-modal for tractability
    rng = np.random.RandomState(42)
    if len(crossmodal_idx) > args.n_features:
        cm_sample = list(rng.choice(crossmodal_idx, args.n_features, replace=False))
    else:
        cm_sample = crossmodal_idx
    mt_sample = matched_idx[:args.n_features]

    # Load SAE
    # Use the scaled SAE (83% var) — prefer the retrained version, fall back to esm3_scaled
    scaled_path = ROOT / "models" / "sae" / "esm3" / f"S_St_layer_{args.layer}_scaled_topk" / "best.pt"
    fallback_path = ROOT / "models" / "sae" / "esm3_scaled" / f"S_St_layer_{args.layer}_topk" / "best.pt"
    old_path = ROOT / "models" / "sae" / "esm3" / f"S_St_layer_{args.layer}_topk" / "best.pt"
    sae_path = scaled_path if scaled_path.exists() else (fallback_path if fallback_path.exists() else old_path)
    log.info(f"Using SAE: {sae_path}")
    sae = load_sae(str(sae_path))
    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / f"S+St_layer_{args.layer}.h5")

    # Characterize cross-modal features
    log.info(f"\nCharacterizing {len(cm_sample)} cross-modal features...")
    cm_chars = characterize_structural_features(
        sae, h5_path, cm_sample, sequences, offsets_dict, accs,
        dssp_data, metadata, structure_dir, n_proteins=args.n_proteins)

    # Characterize matched features
    log.info(f"\nCharacterizing {len(mt_sample)} matched features...")
    mt_chars = characterize_structural_features(
        sae, h5_path, mt_sample, sequences, offsets_dict, accs,
        dssp_data, metadata, structure_dir, n_proteins=args.n_proteins)

    # Compare
    log.info("\n" + "=" * 60)
    log.info("STRUCTURAL COMPARISON: Cross-modal vs Matched")
    log.info("=" * 60)

    comparison = {}
    metrics = [
        ("mean_contact_enrichment", "Contact enrichment"),
        ("mean_spatial_clustering", "Spatial clustering (lower=more clustered)"),
        ("mean_ss_transition_frac", "SS transition fraction"),
        ("mean_burial_frac", "Burial fraction"),
        ("mean_surface_frac", "Surface fraction"),
        ("mean_loop_frac", "Loop/coil fraction"),
    ]

    for metric, desc in metrics:
        cm_vals = [c[metric] for c in cm_chars if c[metric] is not None]
        mt_vals = [c[metric] for c in mt_chars if c[metric] is not None]

        if cm_vals and mt_vals:
            u_stat, u_p = stats.mannwhitneyu(cm_vals, mt_vals, alternative="two-sided")
            log.info(f"\n{desc}:")
            log.info(f"  Cross-modal: {np.mean(cm_vals):.3f} +/- {np.std(cm_vals):.3f} (n={len(cm_vals)})")
            log.info(f"  Matched:     {np.mean(mt_vals):.3f} +/- {np.std(mt_vals):.3f} (n={len(mt_vals)})")
            log.info(f"  U-test p={u_p:.3g}")
            comparison[metric] = {
                "crossmodal_mean": float(np.mean(cm_vals)),
                "crossmodal_std": float(np.std(cm_vals)),
                "crossmodal_n": len(cm_vals),
                "matched_mean": float(np.mean(mt_vals)),
                "matched_std": float(np.std(mt_vals)),
                "matched_n": len(mt_vals),
                "U": float(u_stat),
                "p": float(u_p),
            }
        else:
            log.info(f"\n{desc}: insufficient data (cm={len(cm_vals)}, mt={len(mt_vals)})")

    # Save
    output = {
        "layer": args.layer,
        "n_crossmodal_characterized": len(cm_chars),
        "n_matched_characterized": len(mt_chars),
        "n_proteins": args.n_proteins,
        "crossmodal_features": cm_chars,
        "matched_features": mt_chars,
        "comparison": comparison,
    }

    out_path = out_dir / f"crossmodal_characterization_L{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
