#!/usr/bin/env python3
"""Interpretable probing: which SAE features drive functional site prediction?

Following Adams et al. (ICML 2025): train a linear probe on SAE features
for functional site detection, then inspect which features have the highest
probe weights. Compare cross-modal vs matched feature recruitment.

Usage:
    CUDA_VISIBLE_DEVICES=0 ./env/bin/python scripts/unified/run_interpretable_probing.py
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
from scipy import stats, sparse

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("interp_probe")


def load_sae(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sae = TopKSAE(ckpt["sae_config"])
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-proteins", type=int, default=3000,
                        help="Number of proteins for training")
    parser.add_argument("--n-test-proteins", type=int, default=500,
                        help="Number of proteins for testing")
    parser.add_argument("--max-residues", type=int, default=500000,
                        help="Max residues for training (subsample if larger)")
    args = parser.parse_args()

    # Load cross-modal/matched indices
    cm_path = ROOT / "results" / "unified" / "crossmodal_features" / "crossmodal_features_L33.json"
    with open(cm_path) as f:
        cm_data = json.load(f)
    crossmodal_idx = set(cm_data["crossmodal_feature_indices"])
    matched_idx = set(cm_data["matched_feature_indices"])
    alive_features = crossmodal_idx | matched_idx
    log.info(f"Cross-modal: {len(crossmodal_idx)}, Matched: {len(matched_idx)}, "
             f"Alive: {len(alive_features)}")

    # Load SAE
    sae_path = ROOT / "models" / "sae" / "esm3" / "S_St_layer_33_scaled_topk" / "best.pt"
    sae = load_sae(str(sae_path))
    d_sae = sae.config.input_dim * sae.config.expansion_factor
    log.info(f"SAE dimension: {d_sae}")

    h5_path = str(ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S+St_layer_33.h5")

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    all_accs = sorted(sequences.keys())
    offsets = {}
    pos = 0
    for acc in all_accs:
        L = len(sequences[acc])
        offsets[acc] = (pos, L)
        pos += L

    # Split train/test
    train_accs = all_accs[:args.n_train_proteins]
    test_accs = all_accs[args.n_train_proteins:args.n_train_proteins + args.n_test_proteins]
    log.info(f"Train proteins: {len(train_accs)}, Test proteins: {len(test_accs)}")

    # ═══════════════════════════════════════════════════
    # Build feature matrices and labels
    # ═══════════════════════════════════════════════════
    def build_dataset(accs, max_residues=None, desc=""):
        """Build sparse SAE feature matrix and functional site labels."""
        rows_data = []  # list of (row_indices, col_indices, values) for sparse matrix
        labels = []
        row_idx = 0

        with h5py.File(h5_path, "r") as f:
            acts_data = f["activations"]
            total_residues = acts_data.shape[0]

            for pi, acc in enumerate(accs):
                if acc not in offsets:
                    continue
                start, L = offsets[acc]
                if start + L > total_residues or L < 5:
                    continue

                # Get functional sites
                meta = metadata.get(acc, {})
                func_sites = set()
                for feat in meta.get("features", []):
                    for p in range(feat["start"] - 1, feat["end"]):
                        if 0 <= p < L:
                            func_sites.add(p)

                # Encode through SAE
                raw = torch.tensor(acts_data[start:start + L], dtype=torch.float32)
                with torch.no_grad():
                    z = sae.encode(raw).numpy()  # (L, d_sae)

                # Store sparse data
                for pos_i in range(L):
                    nonzero = np.where(z[pos_i] > 0)[0]
                    for col in nonzero:
                        rows_data.append((row_idx, int(col), float(z[pos_i, col])))
                    labels.append(1 if pos_i in func_sites else 0)
                    row_idx += 1

                if (pi + 1) % 500 == 0:
                    log.info(f"  {desc}: {pi + 1}/{len(accs)} proteins, {row_idx} residues")

                if max_residues and row_idx >= max_residues:
                    break

        log.info(f"  {desc}: {row_idx} residues, {sum(labels)} functional sites "
                 f"({100*sum(labels)/max(len(labels),1):.1f}%)")

        # Build sparse matrix
        if not rows_data:
            return None, None

        r_idx = [d[0] for d in rows_data]
        c_idx = [d[1] for d in rows_data]
        vals = [d[2] for d in rows_data]

        X = sparse.csr_matrix((vals, (r_idx, c_idx)),
                              shape=(row_idx, d_sae))
        y = np.array(labels[:row_idx])

        return X, y

    log.info("\nBuilding training set...")
    X_train, y_train = build_dataset(train_accs, max_residues=args.max_residues, desc="Train")

    log.info("\nBuilding test set...")
    X_test, y_test = build_dataset(test_accs, max_residues=100000, desc="Test")

    if X_train is None or X_test is None:
        log.error("Failed to build datasets")
        return

    log.info(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
    log.info(f"Train positive rate: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

    # ═══════════════════════════════════════════════════
    # Also build raw ESM feature matrices for baseline
    # ═══════════════════════════════════════════════════
    log.info("\nBuilding raw ESM baseline datasets...")

    def build_raw_dataset(accs, max_residues=None, desc=""):
        """Build raw ESM activation matrix and functional site labels."""
        X_rows = []
        labels = []
        with h5py.File(h5_path, "r") as f:
            acts_data = f["activations"]
            total_residues = acts_data.shape[0]
            for pi, acc in enumerate(accs):
                if acc not in offsets:
                    continue
                start, L = offsets[acc]
                if start + L > total_residues or L < 5:
                    continue
                meta = metadata.get(acc, {})
                func_sites = set()
                for feat in meta.get("features", []):
                    for p in range(feat["start"] - 1, feat["end"]):
                        if 0 <= p < L:
                            func_sites.add(p)
                raw = acts_data[start:start + L]
                X_rows.append(raw)
                labels.extend([1 if p in func_sites else 0 for p in range(L)])
                if max_residues and len(labels) >= max_residues:
                    break
        if not X_rows:
            return None, None
        X = np.vstack(X_rows)[:len(labels)]
        y = np.array(labels[:X.shape[0]])
        log.info(f"  {desc}: {X.shape[0]} residues, {y.sum()} functional ({100*y.mean():.1f}%)")
        return X, y

    X_train_raw, _ = build_raw_dataset(train_accs, max_residues=args.max_residues, desc="Raw train")
    X_test_raw, _ = build_raw_dataset(test_accs, max_residues=100000, desc="Raw test")

    # ═══════════════════════════════════════════════════
    # Train probes with CV for hyperparameter selection
    # ═══════════════════════════════════════════════════
    log.info("\nTraining probes with hyperparameter search...")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, classification_report

    # Split train into train/val for C selection (80/20)
    n_val_split = int(0.2 * X_train.shape[0])
    idx = np.random.RandomState(42).permutation(X_train.shape[0])
    val_idx = idx[:n_val_split]
    train_idx = idx[n_val_split:]

    X_tr = X_train[train_idx]
    y_tr = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]

    # Search over C values for SAE probe (liblinear is much faster than saga for sparse L1)
    best_c, best_auroc = None, -1
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        log.info(f"  Trying C={C}...")
        clf_cv = LogisticRegression(
            penalty="l1", C=C, solver="liblinear", max_iter=500,
            class_weight="balanced", random_state=42,
        )
        clf_cv.fit(X_tr, y_tr)
        val_proba = clf_cv.decision_function(X_val)
        auroc = roc_auc_score(y_val, val_proba)
        log.info(f"    Val AUROC = {auroc:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_c = C

    log.info(f"\nBest C={best_c} (val AUROC={best_auroc:.4f})")

    # Retrain on full training set with best C
    clf = LogisticRegression(
        penalty="l1", C=best_c, solver="liblinear", max_iter=1000,
        class_weight="balanced", random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred_proba = clf.decision_function(X_test)
    y_pred = clf.predict(X_test)

    # Train raw ESM baseline probe
    log.info("\nTraining raw ESM baseline probe...")
    raw_auroc = None
    best_c_raw = 0.1
    if X_train_raw is not None and X_test_raw is not None:
        n_raw = min(X_train_raw.shape[0], len(y_train))
        n_raw_tr = int(0.8 * n_raw)
        y_raw_train = y_train[:n_raw]

        best_auroc_raw = -1
        for C in [0.01, 0.1, 1.0]:
            clf_raw_cv = LogisticRegression(
                penalty="l1", C=C, solver="liblinear", max_iter=300,
                class_weight="balanced", random_state=42,
            )
            clf_raw_cv.fit(X_train_raw[:n_raw_tr], y_raw_train[:n_raw_tr])
            val_proba_raw = clf_raw_cv.decision_function(X_train_raw[n_raw_tr:n_raw])
            auroc_raw_val = roc_auc_score(y_raw_train[n_raw_tr:n_raw], val_proba_raw)
            log.info(f"  Raw C={C}: val AUROC={auroc_raw_val:.4f}")
            if auroc_raw_val > best_auroc_raw:
                best_auroc_raw = auroc_raw_val
                best_c_raw = C

        clf_raw = LogisticRegression(
            penalty="l1", C=best_c_raw, solver="liblinear", max_iter=500,
            class_weight="balanced", random_state=42,
        )
        clf_raw.fit(X_train_raw[:n_raw], y_raw_train)
        n_raw_test = min(X_test_raw.shape[0], len(y_test))
        raw_pred_proba = clf_raw.decision_function(X_test_raw[:n_raw_test])
        raw_auroc = float(roc_auc_score(y_test[:n_raw_test], raw_pred_proba))
        log.info(f"Raw ESM baseline AUROC: {raw_auroc:.4f}")
    else:
        log.warning("Could not build raw ESM baseline")

    auroc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    log.info(f"Test AUROC: {auroc:.4f}")
    log.info(f"Test accuracy: {report['accuracy']:.4f}")
    log.info(f"Test precision (functional): {report.get('1', {}).get('precision', 0):.4f}")
    log.info(f"Test recall (functional): {report.get('1', {}).get('recall', 0):.4f}")

    # ═══════════════════════════════════════════════════
    # Analyze probe weights
    # ═══════════════════════════════════════════════════
    log.info("\nAnalyzing probe weights...")

    coefs = clf.coef_[0]  # shape (d_sae,)
    abs_coefs = np.abs(coefs)

    # Rank features by |coefficient|
    ranked_idx = np.argsort(-abs_coefs)

    # Only consider alive features (non-dead SAE features)
    alive_ranked = [int(idx) for idx in ranked_idx if idx in alive_features]

    # Top-N analysis
    for top_n in [20, 50, 100, 200]:
        top_features = alive_ranked[:top_n]
        n_cm = sum(1 for f in top_features if f in crossmodal_idx)
        n_mt = sum(1 for f in top_features if f in matched_idx)
        frac_cm = n_cm / max(top_n, 1)

        # Expected fraction (background rate)
        bg_cm_frac = len(crossmodal_idx) / len(alive_features)

        # Hypergeometric test: is cross-modal overrepresented?
        # k = n_cm successes, M = total alive, n = total CM, N = top_n drawn
        hyper_p = stats.hypergeom.sf(n_cm - 1, len(alive_features),
                                      len(crossmodal_idx), min(top_n, len(alive_features)))

        log.info(f"\n  Top-{top_n} features: CM={n_cm} ({100*frac_cm:.1f}%), "
                 f"MT={n_mt} ({100*n_mt/max(top_n,1):.1f}%) | "
                 f"Background CM: {100*bg_cm_frac:.1f}% | "
                 f"Hypergeometric p={hyper_p:.4f}")

    # Weight distribution: cross-modal vs matched
    cm_weights = [abs_coefs[f] for f in crossmodal_idx if abs_coefs[f] > 0]
    mt_weights = [abs_coefs[f] for f in matched_idx if abs_coefs[f] > 0]

    log.info(f"\nNonzero weight features:")
    log.info(f"  Cross-modal: {len(cm_weights)} features, "
             f"mean |w| = {np.mean(cm_weights):.6f}" if cm_weights else "  Cross-modal: 0")
    log.info(f"  Matched:     {len(mt_weights)} features, "
             f"mean |w| = {np.mean(mt_weights):.6f}" if mt_weights else "  Matched: 0")

    if cm_weights and mt_weights:
        stat, wp = stats.mannwhitneyu(cm_weights, mt_weights, alternative="two-sided")
        log.info(f"  Mann-Whitney (|weight|): U={stat:.0f}, p={wp:.4f}")

    # Sign analysis: positive weights = feature predicts functional site
    cm_pos = sum(1 for f in crossmodal_idx if coefs[f] > 0)
    cm_neg = sum(1 for f in crossmodal_idx if coefs[f] < 0)
    mt_pos = sum(1 for f in matched_idx if coefs[f] > 0)
    mt_neg = sum(1 for f in matched_idx if coefs[f] < 0)

    log.info(f"\nWeight sign distribution:")
    log.info(f"  Cross-modal: {cm_pos} positive, {cm_neg} negative")
    log.info(f"  Matched:     {mt_pos} positive, {mt_neg} negative")

    # Top-20 features detail
    log.info(f"\nTop 20 probe features (by |weight|):")
    log.info(f"  {'Rank':>4} {'Feature':>8} {'Weight':>10} {'|Weight|':>10} {'Label':>12}")
    for rank, fid in enumerate(alive_ranked[:20]):
        label = "cross-modal" if fid in crossmodal_idx else "matched"
        log.info(f"  {rank+1:>4} f/{fid:<8} {coefs[fid]:>10.6f} {abs_coefs[fid]:>10.6f} {label:>12}")

    # ═══════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════
    bg_cm_frac = len(crossmodal_idx) / len(alive_features)

    top_features_detail = []
    for rank, fid in enumerate(alive_ranked[:200]):
        label = "cross-modal" if fid in crossmodal_idx else "matched"
        top_features_detail.append({
            "rank": rank + 1,
            "feature_id": int(fid),
            "weight": float(coefs[fid]),
            "abs_weight": float(abs_coefs[fid]),
            "label": label,
        })

    # Enrichment at multiple top-N
    enrichment_analysis = {}
    for top_n in [10, 20, 50, 100, 200]:
        top_feats = alive_ranked[:top_n]
        n_cm = sum(1 for f in top_feats if f in crossmodal_idx)
        hyper_p = stats.hypergeom.sf(n_cm - 1, len(alive_features),
                                      len(crossmodal_idx), min(top_n, len(alive_features)))
        enrichment_analysis[f"top_{top_n}"] = {
            "n_crossmodal": n_cm,
            "n_matched": top_n - n_cm,
            "frac_crossmodal": float(n_cm / max(top_n, 1)),
            "background_frac_crossmodal": float(bg_cm_frac),
            "enrichment_ratio": float((n_cm / max(top_n, 1)) / bg_cm_frac) if bg_cm_frac > 0 else 0,
            "hypergeometric_p": float(hyper_p),
        }

    output = {
        "probe_performance": {
            "auroc": float(auroc),
            "raw_esm_auroc": raw_auroc,
            "best_C": float(best_c),
            "best_C_raw": float(best_c_raw),
            "accuracy": float(report["accuracy"]),
            "precision_functional": float(report.get("1", {}).get("precision", 0)),
            "recall_functional": float(report.get("1", {}).get("recall", 0)),
            "f1_functional": float(report.get("1", {}).get("f1-score", 0)),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "weight_analysis": {
            "n_nonzero_cm": len(cm_weights),
            "n_nonzero_mt": len(mt_weights),
            "mean_abs_weight_cm": float(np.mean(cm_weights)) if cm_weights else 0,
            "mean_abs_weight_mt": float(np.mean(mt_weights)) if mt_weights else 0,
            "weight_mannwhitney_p": float(wp) if cm_weights and mt_weights else None,
            "cm_positive": cm_pos, "cm_negative": cm_neg,
            "mt_positive": mt_pos, "mt_negative": mt_neg,
        },
        "enrichment_analysis": enrichment_analysis,
        "top_features": top_features_detail,
        "background_crossmodal_frac": float(bg_cm_frac),
    }

    out_path = ROOT / "results" / "unified" / "crossmodal_features" / "interpretable_probing_L33.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
