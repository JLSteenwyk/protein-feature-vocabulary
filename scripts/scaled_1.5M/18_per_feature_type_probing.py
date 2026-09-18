#!/usr/bin/env python3
"""Per-feature-type probing breakdown: separate AUROC for each functional site type.

Evaluates the SAE probe separately on active sites, binding sites, metal binding,
transmembrane regions, etc.

Usage:
    ./env/bin/python scripts/scaled_1.5M/18_per_feature_type_probing.py
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import h5py
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feat_type_probe")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

FEATURE_TYPES = [
    "Active site",
    "Binding site",
    "Metal binding",
    "Site",
    "Transmembrane",
    "Signal peptide",
    "Modified residue",
    "Disulfide bond",
]

MAX_TRAIN = 500_000
MAX_TEST = 200_000


def run_per_type_probing(model_name, metadata):
    """Train one probe per functional site type."""
    log.info(f"\n{'='*60}")
    log.info(f"PER-TYPE PROBING: {model_name}")
    log.info(f"{'='*60}")

    feat_dir = FEATURE_ROOT / model_name / "residue"
    X_sparse = sparse.load_npz(str(feat_dir / "features_sparse.npz"))

    with h5py.File(feat_dir / "protein_summaries.h5", "r") as f:
        protein_ids = [s.decode() for s in f["protein_ids"][:]]
        offsets = f["offsets"][:]

    log.info(f"  {X_sparse.shape[0]} residues, {X_sparse.shape[1]} features")

    # Train/test split (same as 09)
    rng = np.random.RandomState(42)
    n_proteins = len(protein_ids)
    perm = rng.permutation(n_proteins)
    n_train = int(0.8 * n_proteins)
    train_prot_idx = set(perm[:n_train])

    train_residue_idx = []
    test_residue_idx = []
    train_protein_map = {}  # residue_idx -> (protein_idx, local_pos)
    test_protein_map = {}

    for pi in range(n_proteins):
        start, L = offsets[pi]
        for pos in range(L):
            ridx = start + pos
            if pi in train_prot_idx:
                train_residue_idx.append(ridx)
                train_protein_map[ridx] = (pi, pos)
            else:
                test_residue_idx.append(ridx)
                test_protein_map[ridx] = (pi, pos)

    train_residue_idx = np.array(train_residue_idx)
    test_residue_idx = np.array(test_residue_idx)

    results_per_type = {}

    for feat_type in FEATURE_TYPES:
        # Build labels for this feature type only
        y_all = np.zeros(X_sparse.shape[0], dtype=np.int32)
        n_annotated_proteins = 0

        for pi, pid in enumerate(protein_ids):
            start, L = offsets[pi]
            meta = metadata.get(pid, {})
            has_type = False
            for feat in meta.get("features", []):
                if feat["type"] == feat_type:
                    has_type = True
                    for pos in range(feat["start"] - 1, feat["end"]):
                        if 0 <= pos < L:
                            y_all[start + pos] = 1
            if has_type:
                n_annotated_proteins += 1

        y_train = y_all[train_residue_idx]
        y_test = y_all[test_residue_idx]

        n_pos_train = y_train.sum()
        n_pos_test = y_test.sum()

        log.info(f"\n  {feat_type}:")
        log.info(f"    Train: {n_pos_train} positive / {len(y_train)} total")
        log.info(f"    Test:  {n_pos_test} positive / {len(y_test)} total")
        log.info(f"    Proteins with annotation: {n_annotated_proteins}")

        if n_pos_train < 20 or n_pos_test < 10:
            log.info(f"    Skipping (too few positives)")
            results_per_type[feat_type] = {
                "n_pos_train": int(n_pos_train),
                "n_pos_test": int(n_pos_test),
                "skipped": True,
            }
            continue

        # Subsample for training
        X_tr = X_sparse[train_residue_idx]
        X_te = X_sparse[test_residue_idx]

        if len(y_train) > MAX_TRAIN:
            idx = rng.choice(len(y_train), MAX_TRAIN, replace=False)
            X_tr_sub = X_tr[idx]
            y_tr_sub = y_train[idx]
        else:
            X_tr_sub = X_tr
            y_tr_sub = y_train

        if len(y_test) > MAX_TEST:
            idx = rng.choice(len(y_test), MAX_TEST, replace=False)
            X_te_sub = X_te[idx]
            y_te_sub = y_test[idx]
        else:
            X_te_sub = X_te
            y_te_sub = y_test

        # Quick C search
        best_c, best_auroc = 0.001, 0
        n_val = int(0.2 * X_tr_sub.shape[0])
        val_mask = np.zeros(X_tr_sub.shape[0], dtype=bool)
        val_mask[rng.choice(X_tr_sub.shape[0], n_val, replace=False)] = True

        for C in [0.0001, 0.001, 0.01]:
            try:
                clf = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                         max_iter=300, class_weight="balanced", random_state=42)
                clf.fit(X_tr_sub[~val_mask], y_tr_sub[~val_mask])
                pred = clf.decision_function(X_tr_sub[val_mask])
                if len(np.unique(y_tr_sub[val_mask])) < 2:
                    continue
                auroc = roc_auc_score(y_tr_sub[val_mask], pred)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_c = C
            except Exception:
                continue

        # Final model
        try:
            clf = LogisticRegression(penalty="l1", C=best_c, solver="liblinear",
                                     max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_tr_sub, y_tr_sub)
            test_pred = clf.decision_function(X_te_sub)
            test_auroc = roc_auc_score(y_te_sub, test_pred)

            n_nonzero = (np.abs(clf.coef_[0]) > 0).sum()

            log.info(f"    AUROC: {test_auroc:.4f} (C={best_c}, {n_nonzero} nonzero weights)")

            results_per_type[feat_type] = {
                "test_auroc": float(test_auroc),
                "best_C": float(best_c),
                "n_pos_train": int(n_pos_train),
                "n_pos_test": int(n_pos_test),
                "n_annotated_proteins": n_annotated_proteins,
                "n_nonzero_weights": int(n_nonzero),
                "skipped": False,
            }
        except Exception as e:
            log.warning(f"    Error: {e}")
            results_per_type[feat_type] = {"skipped": True, "error": str(e)}

    return {"model": model_name, "per_type": results_per_type}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    # Count feature types in eval set
    type_counts = Counter()
    for m in metadata.values():
        for feat in m.get("features", []):
            type_counts[feat["type"]] += 1
    log.info("Feature type counts in eval set:")
    for ft, n in type_counts.most_common():
        log.info(f"  {ft:25s}: {n:>6d}")

    results = {}
    for model_name in ["esm3", "esm2"]:
        feat_path = FEATURE_ROOT / model_name / "residue" / "features_sparse.npz"
        if not feat_path.exists():
            continue
        results[model_name] = run_per_type_probing(model_name, metadata)

    out_path = OUTPUT_DIR / "per_feature_type_probing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
