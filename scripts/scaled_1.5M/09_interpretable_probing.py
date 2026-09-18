#!/usr/bin/env python3
"""Interpretable probing: which SAE features drive functional site prediction?

Uses pre-computed sparse SAE feature matrices from 08_encode_eval_through_saes.py.
Trains L1 logistic regression (liblinear) on SAE features for functional site detection.
Compares SAE probe vs raw ESM activation baseline.

Usage:
    ./env/bin/python scripts/scaled_1.5M/09_interpretable_probing.py
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("probing")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
ACT_ROOT = ROOT / "results" / "scaled_1.5M" / "activations"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"


def build_labels(protein_ids, offsets, metadata):
    """Build per-residue functional site labels."""
    labels = np.zeros(sum(o[1] for o in offsets), dtype=np.int32)

    for pi, pid in enumerate(protein_ids):
        start, L = offsets[pi]
        meta = metadata.get(pid, {})
        for feat in meta.get("features", []):
            if feat["type"] in ["Active site", "Binding site", "Metal binding", "Site"]:
                for pos in range(feat["start"] - 1, feat["end"]):
                    if 0 <= pos < L:
                        labels[start + pos] = 1
    return labels


def run_probing(model_name, metadata):
    """Run interpretable probing for one model."""
    log.info(f"\n{'='*60}")
    log.info(f"INTERPRETABLE PROBING: {model_name}")
    log.info(f"{'='*60}")

    # Load sparse features
    feat_dir = FEATURE_ROOT / model_name / "residue"
    X_sparse = sparse.load_npz(str(feat_dir / "features_sparse.npz"))

    with h5py.File(feat_dir / "protein_summaries.h5", "r") as f:
        protein_ids = [s.decode() for s in f["protein_ids"][:]]
        offsets = f["offsets"][:]

    log.info(f"  Feature matrix: {X_sparse.shape}, nnz={X_sparse.nnz:,}")

    # Build labels
    y = build_labels(protein_ids, offsets, metadata)
    n_pos = y.sum()
    log.info(f"  Labels: {len(y)} residues, {n_pos} functional sites ({100*n_pos/len(y):.1f}%)")

    if n_pos < 50:
        log.warning(f"  Too few functional sites, skipping {model_name}")
        return None

    # Train/test split by protein (80/20)
    rng = np.random.RandomState(42)
    n_proteins = len(protein_ids)
    perm = rng.permutation(n_proteins)
    n_train = int(0.8 * n_proteins)
    train_prot_idx = set(perm[:n_train])

    # Build train/test residue indices
    train_residue_idx = []
    test_residue_idx = []
    for pi in range(n_proteins):
        start, L = offsets[pi]
        idx = list(range(start, start + L))
        if pi in train_prot_idx:
            train_residue_idx.extend(idx)
        else:
            test_residue_idx.extend(idx)

    train_residue_idx = np.array(train_residue_idx)
    test_residue_idx = np.array(test_residue_idx)

    X_train = X_sparse[train_residue_idx]
    y_train = y[train_residue_idx]
    X_test = X_sparse[test_residue_idx]
    y_test = y[test_residue_idx]

    log.info(f"  Train: {X_train.shape[0]} residues, {y_train.sum()} pos ({100*y_train.mean():.1f}%)")
    log.info(f"  Test:  {X_test.shape[0]} residues, {y_test.sum()} pos ({100*y_test.mean():.1f}%)")

    # Protein-grouped CV for C selection (avoids leaking residues from same protein)
    MAX_CV_RESIDUES_PER_FOLD = 500_000
    N_CV_FOLDS = 3

    # Build protein-level fold assignments for training proteins only
    train_prot_list = sorted(train_prot_idx)  # deterministic order
    rng.shuffle(train_prot_list)              # shuffle with seeded rng
    prot_fold = {}  # protein_index -> fold_id
    for i, pi in enumerate(train_prot_list):
        prot_fold[pi] = i % N_CV_FOLDS

    # Pre-compute residue indices per protein (within the training set)
    # Map: protein_index -> list of positions in X_train (0-based into X_train)
    prot_to_train_residues = {}
    cursor = 0
    for pi in range(n_proteins):
        _, L = offsets[pi]
        if pi in train_prot_idx:
            prot_to_train_residues[pi] = list(range(cursor, cursor + L))
            cursor += L

    log.info(f"  Protein-grouped {N_CV_FOLDS}-fold CV for C selection "
             f"({len(train_prot_list)} train proteins)")

    best_c, best_auroc = None, -1
    cv_results = {}
    for C in [0.0001, 0.001, 0.01, 0.1]:
        fold_aurocs = []
        for fold in range(N_CV_FOLDS):
            # Split proteins into CV-train and CV-val by fold
            cv_train_residues = []
            cv_val_residues = []
            for pi in train_prot_list:
                residues = prot_to_train_residues[pi]
                if prot_fold[pi] == fold:
                    cv_val_residues.extend(residues)
                else:
                    cv_train_residues.extend(residues)

            cv_train_residues = np.array(cv_train_residues)
            cv_val_residues = np.array(cv_val_residues)

            # Subsample within each split if needed for speed
            if len(cv_train_residues) > MAX_CV_RESIDUES_PER_FOLD:
                cv_train_residues = rng.choice(
                    cv_train_residues, MAX_CV_RESIDUES_PER_FOLD, replace=False)
            if len(cv_val_residues) > MAX_CV_RESIDUES_PER_FOLD:
                cv_val_residues = rng.choice(
                    cv_val_residues, MAX_CV_RESIDUES_PER_FOLD, replace=False)

            X_cv_tr = X_train[cv_train_residues]
            y_cv_tr = y_train[cv_train_residues]
            X_cv_val = X_train[cv_val_residues]
            y_cv_val = y_train[cv_val_residues]

            # Skip fold if val set has no positive labels
            if y_cv_val.sum() == 0:
                log.warning(f"    C={C} fold={fold}: no positive labels in val, skipping")
                continue

            clf = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                     max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_cv_tr, y_cv_tr)
            val_proba = clf.decision_function(X_cv_val)
            auroc = roc_auc_score(y_cv_val, val_proba)
            fold_aurocs.append(auroc)

        if fold_aurocs:
            mean_auroc = np.mean(fold_aurocs)
        else:
            mean_auroc = -1
        log.info(f"    C={C}: mean AUROC={mean_auroc:.4f} "
                 f"(folds: {[f'{a:.4f}' for a in fold_aurocs]})")
        cv_results[C] = float(mean_auroc)
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            best_c = C

    log.info(f"  Best C={best_c} (mean CV AUROC={best_auroc:.4f})")

    # Final model trained on full training set (~1M residues, comparable to Adams et al.)
    MAX_TRAIN_RESIDUES = 1_000_000
    if X_train.shape[0] > MAX_TRAIN_RESIDUES:
        final_idx = rng.choice(X_train.shape[0], MAX_TRAIN_RESIDUES, replace=False)
        X_final = X_train[final_idx]
        y_final = y_train[final_idx]
        log.info(f"  Training final model on {MAX_TRAIN_RESIDUES} residues")
    else:
        X_final = X_train
        y_final = y_train

    clf = LogisticRegression(penalty="l1", C=best_c, solver="liblinear",
                             max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_final, y_final)
    test_proba = clf.decision_function(X_test)
    test_auroc = roc_auc_score(y_test, test_proba)
    report = classification_report(y_test, clf.predict(X_test), output_dict=True, zero_division=0)

    log.info(f"  Test AUROC: {test_auroc:.4f}")
    log.info(f"  Precision: {report.get('1', {}).get('precision', 0):.4f}")
    log.info(f"  Recall: {report.get('1', {}).get('recall', 0):.4f}")

    # Analyze probe weights
    coefs = clf.coef_[0]
    abs_coefs = np.abs(coefs)
    nonzero_features = np.where(abs_coefs > 0)[0]
    log.info(f"  Nonzero probe weights: {len(nonzero_features)}/{len(coefs)}")

    # Top features
    ranked = np.argsort(-abs_coefs)
    top_features = []
    for rank, fid in enumerate(ranked[:50]):
        top_features.append({
            "rank": rank + 1,
            "feature_id": int(fid),
            "weight": float(coefs[fid]),
            "abs_weight": float(abs_coefs[fid]),
        })

    return {
        "model": model_name,
        "test_auroc": float(test_auroc),
        "best_C": float(best_c),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
        "precision": float(report.get("1", {}).get("precision", 0)),
        "recall": float(report.get("1", {}).get("recall", 0)),
        "f1": float(report.get("1", {}).get("f1-score", 0)),
        "n_nonzero_weights": int(len(nonzero_features)),
        "top_features": top_features,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    results = {}
    for model_name in ["esm3", "esm2"]:
        feat_path = FEATURE_ROOT / model_name / "residue" / "features_sparse.npz"
        if not feat_path.exists():
            log.warning(f"No features for {model_name}, skipping")
            continue
        result = run_probing(model_name, metadata)
        if result:
            results[model_name] = result

    out_path = OUTPUT_DIR / "interpretable_probing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")

    # Summary
    for model, r in results.items():
        log.info(f"  {model}: AUROC={r['test_auroc']:.4f}, "
                 f"nonzero weights={r['n_nonzero_weights']}")


if __name__ == "__main__":
    main()
