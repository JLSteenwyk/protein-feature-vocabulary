#!/usr/bin/env python3
"""Discover cross-modal SAE features: present in S+St but absent in S-only.

These features represent structural knowledge that the model can only learn
when given explicit structure tokens — invisible to sequence analysis alone.

Approach:
1. Load S-only and S+St SAEs at the same layer
2. Encode a shared set of residues through both SAEs (each on its own activations)
3. Match S+St features to S-only features via activation-pattern correlation
   (features detecting the same biology fire at the same positions)
4. Features with no good S-only match are "cross-modal"
5. Characterize cross-modal vs matched features by:
   - SS preference, surface/buried, functional sites
   - AA specificity, activation statistics
6. Statistical tests for differences

Usage:
    ./env/bin/python scripts/unified/run_crossmodal_feature_discovery.py --layer 33
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

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("crossmodal")


def load_sae(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def build_offsets(sequences):
    accs = sorted(sequences.keys())
    offsets = {}
    idx = 0
    for acc in accs:
        offsets[acc] = (idx, len(sequences[acc]))
        idx += len(sequences[acc])
    return offsets, accs


def match_features_by_activation(sae_sst, sae_s, h5_sst_path, h5_s_path,
                                  alive_sst, alive_s,
                                  n_residues=50000, batch_size=4096):
    """Match S+St features to S-only features by activation-pattern correlation.

    For each feature, compute a binary activation vector over shared residue positions.
    Then compute correlation between S+St and S-only activation vectors.
    Features detecting the same biology fire at the same positions regardless of modality.
    """
    alive_sst_list = sorted(alive_sst)
    alive_s_list = sorted(alive_s)

    log.info(f"  Encoding {n_residues} residues through both SAEs...")

    # Encode through S+St SAE
    z_sst_all = []
    with h5py.File(h5_sst_path, "r") as f:
        acts = f["activations"]
        total = min(acts.shape[0], n_residues)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            raw = torch.tensor(acts[start:end], dtype=torch.float32)
            with torch.no_grad():
                z = sae_sst.encode(raw)
            # Only keep alive features
            z_sst_all.append(z[:, alive_sst_list].numpy())
    z_sst_mat = np.concatenate(z_sst_all, axis=0)  # (N, n_alive_sst)

    # Encode through S-only SAE
    z_s_all = []
    with h5py.File(h5_s_path, "r") as f:
        acts = f["activations"]
        total = min(acts.shape[0], n_residues)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            raw = torch.tensor(acts[start:end], dtype=torch.float32)
            with torch.no_grad():
                z = sae_s.encode(raw)
            z_s_all.append(z[:, alive_s_list].numpy())
    z_s_mat = np.concatenate(z_s_all, axis=0)  # (N, n_alive_s)

    N = min(z_sst_mat.shape[0], z_s_mat.shape[0])
    z_sst_mat = z_sst_mat[:N]
    z_s_mat = z_s_mat[:N]

    log.info(f"  Activation matrices: S+St={z_sst_mat.shape}, S-only={z_s_mat.shape}")

    # Binarize: active or not (more robust than magnitude for cross-SAE comparison)
    z_sst_bin = (z_sst_mat > 0).astype(np.float32)
    z_s_bin = (z_s_mat > 0).astype(np.float32)

    # Compute correlation in chunks (S+St features × S-only features)
    n_sst = len(alive_sst_list)
    n_s = len(alive_s_list)
    max_corrs = np.zeros(n_sst)
    best_match = np.zeros(n_sst, dtype=int)

    # Standardize columns for Pearson correlation
    def standardize(X):
        mu = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-10] = 1.0
        return (X - mu) / std

    z_sst_std = standardize(z_sst_bin)
    z_s_std = standardize(z_s_bin)

    chunk = 500
    for start in range(0, n_sst, chunk):
        end = min(start + chunk, n_sst)
        # corr = (1/N) * z_sst_std.T @ z_s_std
        corrs = z_sst_std[:, start:end].T @ z_s_std / N  # (chunk, n_s)
        max_corrs[start:end] = corrs.max(axis=1)
        best_match[start:end] = corrs.argmax(axis=1)
        if (start // chunk) % 5 == 0:
            log.info(f"    Correlated {end}/{n_sst} S+St features")

    return max_corrs, best_match, alive_sst_list, alive_s_list


def characterize_features(sae, h5_path, feature_indices, sequences, offsets, accs,
                          dssp_data, metadata, sample_proteins=300):
    """Characterize a set of features by their activation patterns."""
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

    rng = np.random.RandomState(42)
    sample = list(rng.choice(accs, min(sample_proteins, len(accs)), replace=False))

    n_feats = len(feature_indices)
    fi_to_local = {fi: i for i, fi in enumerate(feature_indices)}

    # Track per-feature statistics
    feat_count = np.zeros(n_feats)           # number of active residues
    feat_aa_counts = np.zeros((n_feats, 20)) # weighted by activation
    feat_ss_counts = np.zeros((n_feats, 3))  # H, E, C weighted by activation
    feat_asa_sum = np.zeros(n_feats)         # sum of ASA at active sites (unweighted)
    feat_asa_count = np.zeros(n_feats)       # count for ASA averaging
    feat_func_count = np.zeros(n_feats)
    feat_total_for_func = np.zeros(n_feats)
    feat_max_act = np.zeros(n_feats)
    feat_n_proteins = np.zeros(n_feats, dtype=int)

    # Build functional site sets
    func_sites = {}
    for acc in sample:
        m = metadata.get(acc, {})
        sites = set()
        for feat in m.get("features", []):
            if feat.get("type") in ("Active site", "Binding site", "Disulfide bond"):
                for p in range(feat["start"] - 1, feat.get("end", feat["start"])):
                    sites.add(p)
        if sites:
            func_sites[acc] = sites

    with h5py.File(h5_path, "r") as f:
        acts_ds = f["activations"]
        total = acts_ds.shape[0]

        for pi, acc in enumerate(sample):
            start, L = offsets[acc]
            seq = sequences[acc]
            L = min(len(seq), L)
            if start + L > total:
                continue

            raw = torch.tensor(acts_ds[start:start+L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()  # (L, dict_size)

            dssp = dssp_data.get(acc, [])
            sites = func_sites.get(acc, set())

            for fi in feature_indices:
                li = fi_to_local[fi]
                col = z[:, fi]
                active_mask = col > 0
                active_positions = np.where(active_mask)[0]

                n_act = len(active_positions)
                if n_act == 0:
                    continue

                feat_count[li] += n_act
                feat_n_proteins[li] += 1
                local_max = float(col.max())
                if local_max > feat_max_act[li]:
                    feat_max_act[li] = local_max

                # AA distribution (weighted by activation)
                for pos in active_positions:
                    if pos < len(seq) and seq[pos] in aa_to_idx:
                        feat_aa_counts[li, aa_to_idx[seq[pos]]] += col[pos]

                # SS distribution + ASA
                if dssp and len(dssp) >= L:
                    for pos in active_positions:
                        if pos < len(dssp):
                            ss3 = dssp[pos].get("ss3", "C")
                            ss_idx = {"H": 0, "E": 1}.get(ss3, 2)
                            feat_ss_counts[li, ss_idx] += col[pos]
                            asa = dssp[pos].get("asa", 0)
                            if asa is not None:
                                feat_asa_sum[li] += float(asa)
                                feat_asa_count[li] += 1

                # Functional sites
                for pos in active_positions:
                    feat_total_for_func[li] += 1  # count-based, not activation-weighted
                    if pos in sites:
                        feat_func_count[li] += 1

            if (pi + 1) % 100 == 0:
                log.info(f"  Characterized {pi+1}/{len(sample)} proteins")

    # Compile results
    bg_aa = feat_aa_counts.sum(axis=0)
    bg_aa = bg_aa / (bg_aa.sum() + 1e-10)
    bg_ss = feat_ss_counts.sum(axis=0)
    bg_ss = bg_ss / (bg_ss.sum() + 1e-10)

    results = []
    for i, fi in enumerate(feature_indices):
        if feat_count[i] < 10:
            continue

        # AA enrichment
        aa_dist = feat_aa_counts[i] / (feat_aa_counts[i].sum() + 1e-10)
        aa_enr = aa_dist / (bg_aa + 1e-10)
        top_aa_idx = int(np.argmax(aa_enr))
        top_aa = AA_LIST[top_aa_idx]

        # SS preference
        ss_dist = feat_ss_counts[i]
        ss_total = ss_dist.sum()
        if ss_total > 0:
            ss_norm = ss_dist / ss_total
            ss_enr = ss_norm / (bg_ss + 1e-10)
            ss_pref_idx = int(np.argmax(ss_enr))
            ss_pref = ["H", "E", "C"][ss_pref_idx]
            max_ss_enr = float(ss_enr.max())
        else:
            ss_pref = "?"
            max_ss_enr = 0.0

        # Mean ASA (unweighted average over active residues)
        if feat_asa_count[i] > 0:
            mean_asa = float(feat_asa_sum[i] / feat_asa_count[i])
        else:
            mean_asa = -1.0

        # Functional fraction (count-based)
        func_frac = float(feat_func_count[i] / (feat_total_for_func[i] + 1e-10))

        results.append({
            "feature_idx": int(fi),
            "n_activations": int(feat_count[i]),
            "n_proteins": int(feat_n_proteins[i]),
            "max_activation": float(feat_max_act[i]),
            "top_aa": top_aa,
            "top_aa_enrichment": float(aa_enr[top_aa_idx]),
            "ss_preference": ss_pref,
            "ss_enrichment": float(max_ss_enr),
            "mean_asa": mean_asa,
            "functional_frac": func_frac,
            "surface_buried": "surface" if mean_asa > 40 else (
                "buried" if 0 <= mean_asa < 20 else "intermediate"),
        })

    return results


def make_json_safe(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--match-threshold", type=float, default=0.3,
                        help="Activation-pattern correlation below which a feature is 'cross-modal'")
    parser.add_argument("--sample-proteins", type=int, default=400)
    parser.add_argument("--n-residues", type=int, default=50000,
                        help="Number of residues for activation-pattern matching")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "unified" / "crossmodal_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json") as f:
        dssp_data = json.load(f)
    offsets, accs = build_offsets(sequences)

    # ════════════════════════════════════════════════════
    # Step 1: Load both SAEs
    # ════════════════════════════════════════════════════
    log.info(f"Loading SAEs for layer {args.layer}...")
    sae_s = load_sae(f"models/sae/esm3/S_layer_{args.layer}_topk/best.pt")
    # Use the new SAE trained on scaled S+St activations (83% var, cos=0.98)
    # instead of the old one trained on 46K residues (cos=0.20, var=-320%)
    scaled_sst_path = f"models/sae/esm3/S_St_layer_{args.layer}_scaled_topk/best.pt"
    fallback_sst_path = f"models/sae/esm3/S_St_layer_{args.layer}_topk/best.pt"
    sst_path = scaled_sst_path if Path(scaled_sst_path).exists() else fallback_sst_path
    log.info(f"  S+St SAE: {sst_path}")
    sae_sst = load_sae(sst_path)

    h5_sst = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / f"S+St_layer_{args.layer}.h5")
    h5_s = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / f"S_layer_{args.layer}.h5")

    # ════════════════════════════════════════════════════
    # Step 2: Find alive features in both SAEs
    # ════════════════════════════════════════════════════
    log.info("Scanning for alive features...")
    with h5py.File(h5_sst, "r") as f:
        raw = torch.tensor(f["activations"][:20000], dtype=torch.float32)
        with torch.no_grad():
            z_sst = sae_sst.encode(raw).numpy()
        alive_sst = set(int(x) for x in np.where((z_sst > 0).sum(axis=0) >= 5)[0])

    with h5py.File(h5_s, "r") as f:
        raw = torch.tensor(f["activations"][:20000], dtype=torch.float32)
        with torch.no_grad():
            z_s = sae_s.encode(raw).numpy()
        alive_s = set(int(x) for x in np.where((z_s > 0).sum(axis=0) >= 5)[0])

    log.info(f"Alive features: S-only={len(alive_s)}, S+St={len(alive_sst)}")

    # ════════════════════════════════════════════════════
    # Step 3: Match features by activation-pattern correlation
    # ════════════════════════════════════════════════════
    log.info("Matching S+St features to S-only features by activation patterns...")
    max_corrs, best_match_idx, alive_sst_list, alive_s_list = match_features_by_activation(
        sae_sst, sae_s, h5_sst, h5_s,
        alive_sst, alive_s,
        n_residues=args.n_residues
    )

    # Cross-modal = alive in S+St but poor activation-pattern match to any S-only feature
    crossmodal_mask = max_corrs < args.match_threshold
    crossmodal_features = [alive_sst_list[i] for i in np.where(crossmodal_mask)[0]]
    matched_features = [alive_sst_list[i] for i in np.where(~crossmodal_mask)[0]]

    log.info(f"Cross-modal features (corr < {args.match_threshold}): {len(crossmodal_features)}/{len(alive_sst_list)}")
    log.info(f"Matched features: {len(matched_features)}/{len(alive_sst_list)}")

    # Distribution of match quality
    percentiles = np.percentile(max_corrs, [10, 25, 50, 75, 90])
    log.info(f"Match quality percentiles: p10={percentiles[0]:.3f}, p25={percentiles[1]:.3f}, "
             f"p50={percentiles[2]:.3f}, p75={percentiles[3]:.3f}, p90={percentiles[4]:.3f}")

    # ════════════════════════════════════════════════════
    # Step 4: Characterize cross-modal features
    # ════════════════════════════════════════════════════
    # Cap characterization to keep runtime reasonable
    cm_to_characterize = crossmodal_features[:2000]
    log.info(f"\nCharacterizing {len(cm_to_characterize)} cross-modal features...")
    if cm_to_characterize:
        crossmodal_chars = characterize_features(
            sae_sst, h5_sst, cm_to_characterize,
            sequences, offsets, accs, dssp_data, metadata,
            sample_proteins=args.sample_proteins
        )
    else:
        crossmodal_chars = []

    mt_to_characterize = matched_features[:2000]
    log.info(f"\nCharacterizing {len(mt_to_characterize)} matched features (control)...")
    if mt_to_characterize:
        matched_chars = characterize_features(
            sae_sst, h5_sst, mt_to_characterize,
            sequences, offsets, accs, dssp_data, metadata,
            sample_proteins=args.sample_proteins
        )
    else:
        matched_chars = []

    # ════════════════════════════════════════════════════
    # Step 5: Compare cross-modal vs matched features
    # ════════════════════════════════════════════════════
    log.info("\n" + "=" * 60)
    log.info("COMPARISON: Cross-modal vs Matched features")
    log.info("=" * 60)

    comparison = {}
    for label, chars in [("crossmodal", crossmodal_chars), ("matched", matched_chars)]:
        if not chars:
            continue
        ss_prefs = Counter(c["ss_preference"] for c in chars)
        sb_prefs = Counter(c["surface_buried"] for c in chars)
        mean_ss_enr = float(np.mean([c["ss_enrichment"] for c in chars]))
        mean_func = float(np.mean([c["functional_frac"] for c in chars]))
        asa_vals = [c["mean_asa"] for c in chars if c["mean_asa"] >= 0]
        mean_asa = float(np.mean(asa_vals)) if asa_vals else -1.0
        mean_aa_enr = float(np.mean([c["top_aa_enrichment"] for c in chars]))
        n_func = sum(1 for c in chars if c["functional_frac"] > 0.05)
        n_aa_specific = sum(1 for c in chars if c["top_aa_enrichment"] > 4.0)
        n_ss_specific = sum(1 for c in chars if c["ss_enrichment"] > 1.5)

        log.info(f"\n{label.upper()} ({len(chars)} features):")
        log.info(f"  SS preference: {dict(ss_prefs)}")
        log.info(f"  Surface/buried: {dict(sb_prefs)}")
        log.info(f"  Mean SS enrichment: {mean_ss_enr:.2f}")
        log.info(f"  Mean functional frac: {mean_func:.3f}")
        log.info(f"  Mean ASA: {mean_asa:.1f}")
        log.info(f"  Mean AA enrichment: {mean_aa_enr:.2f}")
        log.info(f"  N functional (>5%): {n_func} ({100*n_func/max(len(chars),1):.1f}%)")
        log.info(f"  N AA-specific (>4x): {n_aa_specific} ({100*n_aa_specific/max(len(chars),1):.1f}%)")
        log.info(f"  N SS-specific (>1.5x): {n_ss_specific} ({100*n_ss_specific/max(len(chars),1):.1f}%)")

        comparison[label] = {
            "n_features": len(chars),
            "ss_preferences": {str(k): int(v) for k, v in ss_prefs.items()},
            "surface_buried": {str(k): int(v) for k, v in sb_prefs.items()},
            "mean_ss_enrichment": mean_ss_enr,
            "mean_functional_frac": mean_func,
            "mean_asa": mean_asa,
            "mean_aa_enrichment": mean_aa_enr,
            "n_functional": n_func,
            "n_aa_specific": n_aa_specific,
            "n_ss_specific": n_ss_specific,
        }

    # Statistical tests
    if crossmodal_chars and matched_chars:
        log.info("\n--- Statistical Tests ---")
        cm_ss = [c["ss_enrichment"] for c in crossmodal_chars]
        mt_ss = [c["ss_enrichment"] for c in matched_chars]
        if cm_ss and mt_ss:
            u_stat, u_p = stats.mannwhitneyu(cm_ss, mt_ss, alternative="two-sided")
            log.info(f"SS enrichment: cross-modal={np.mean(cm_ss):.2f} vs matched={np.mean(mt_ss):.2f} (U-test p={u_p:.3g})")
            comparison["ss_enrichment_test"] = {"U": float(u_stat), "p": float(u_p)}

        cm_func = [c["functional_frac"] for c in crossmodal_chars]
        mt_func = [c["functional_frac"] for c in matched_chars]
        if cm_func and mt_func:
            u_stat, u_p = stats.mannwhitneyu(cm_func, mt_func, alternative="two-sided")
            log.info(f"Functional frac: cross-modal={np.mean(cm_func):.3f} vs matched={np.mean(mt_func):.3f} (U-test p={u_p:.3g})")
            comparison["functional_test"] = {"U": float(u_stat), "p": float(u_p)}

        cm_asa = [c["mean_asa"] for c in crossmodal_chars if c["mean_asa"] >= 0]
        mt_asa = [c["mean_asa"] for c in matched_chars if c["mean_asa"] >= 0]
        if cm_asa and mt_asa:
            u_stat, u_p = stats.mannwhitneyu(cm_asa, mt_asa, alternative="two-sided")
            log.info(f"ASA: cross-modal={np.mean(cm_asa):.1f} vs matched={np.mean(mt_asa):.1f} (U-test p={u_p:.3g})")
            comparison["asa_test"] = {"U": float(u_stat), "p": float(u_p)}

        cm_aa = [c["top_aa_enrichment"] for c in crossmodal_chars]
        mt_aa = [c["top_aa_enrichment"] for c in matched_chars]
        if cm_aa and mt_aa:
            u_stat, u_p = stats.mannwhitneyu(cm_aa, mt_aa, alternative="two-sided")
            log.info(f"AA enrichment: cross-modal={np.mean(cm_aa):.2f} vs matched={np.mean(mt_aa):.2f} (U-test p={u_p:.3g})")
            comparison["aa_enrichment_test"] = {"U": float(u_stat), "p": float(u_p)}

    # ════════════════════════════════════════════════════
    # Save results
    # ════════════════════════════════════════════════════
    output = make_json_safe({
        "layer": args.layer,
        "match_method": "activation_pattern_correlation",
        "match_threshold": args.match_threshold,
        "n_residues_for_matching": args.n_residues,
        "n_alive_s": len(alive_s),
        "n_alive_sst": len(alive_sst),
        "n_crossmodal": len(crossmodal_features),
        "n_matched": len(matched_features),
        "match_quality_percentiles": {
            f"p{p}": float(v) for p, v in zip([10, 25, 50, 75, 90], percentiles)
        },
        "crossmodal_features": crossmodal_chars,
        "matched_features_sample": matched_chars,
        "comparison": comparison,
        "crossmodal_feature_indices": [int(x) for x in crossmodal_features],
        "matched_feature_indices": [int(x) for x in matched_features],
    })

    out_path = out_dir / f"crossmodal_features_L{args.layer}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

    # Print top cross-modal features
    if crossmodal_chars:
        log.info(f"\nTop 15 cross-modal features by activation count:")
        top = sorted(crossmodal_chars, key=lambda x: -x["n_activations"])[:15]
        for c in top:
            log.info(f"  f/{c['feature_idx']}: n={c['n_activations']}, proteins={c['n_proteins']}, "
                    f"SS={c['ss_preference']}({c['ss_enrichment']:.2f}x), "
                    f"AA={c['top_aa']}({c['top_aa_enrichment']:.1f}x), "
                    f"func={c['functional_frac']:.3f}, ASA={c['mean_asa']:.0f}")

    # Print some well-matched features for comparison
    if matched_chars:
        log.info(f"\nTop 10 matched features (strongest S-only correspondence):")
        top_matched = sorted(matched_chars, key=lambda x: -x["n_activations"])[:10]
        for c in top_matched:
            log.info(f"  f/{c['feature_idx']}: n={c['n_activations']}, proteins={c['n_proteins']}, "
                    f"SS={c['ss_preference']}({c['ss_enrichment']:.2f}x), "
                    f"AA={c['top_aa']}({c['top_aa_enrichment']:.1f}x), "
                    f"func={c['functional_frac']:.3f}, ASA={c['mean_asa']:.0f}")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
