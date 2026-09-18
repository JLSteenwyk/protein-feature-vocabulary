#!/usr/bin/env python3
"""Raw ESM activation baseline probe for comparison with SAE features.

Uses identical train/test split as 09_interpretable_probing.py.

Usage:
    ./env/bin/python scripts/scaled_1.5M/14_raw_esm_baseline_probe.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("raw_probe")

ACT_ROOT = ROOT / "results" / "scaled_1.5M" / "activations"
FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"

MAX_CV_RESIDUES = 200_000   # Dense matrices need smaller subsample for speed
MAX_TRAIN_RESIDUES = 500_000


def build_labels(protein_ids, offsets, metadata):
    """Build per-residue functional site labels (identical to 09)."""
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


def run_raw_probe(model_name, metadata):
    """Train and evaluate raw ESM baseline probe."""
    log.info(f"\n{'='*60}")
    log.info(f"RAW ESM BASELINE PROBE: {model_name}")
    log.info(f"{'='*60}")

    # Load raw activations from HDF5 chunks
    residue_dir = ACT_ROOT / model_name / "residue_L33"
    chunks = sorted(residue_dir.glob("*.h5"))
    log.info(f"  Loading from {len(chunks)} chunks...")

    all_acts = []
    all_ids = []
    all_offsets = []
    global_offset = 0

    for ci, cp in enumerate(chunks):
        with h5py.File(cp, "r") as f:
            acts = f["activations"][:].astype(np.float32)
            offsets = f["offsets"][:]
            ids = [s.decode() for s in f["ids"][:]]

        for pi, pid in enumerate(ids):
            _, local_off, L = offsets[pi]
            all_offsets.append((global_offset, L))
            all_ids.append(pid)
            global_offset += L

        all_acts.append(acts)

    X_raw = np.concatenate(all_acts, axis=0)
    log.info(f"  Raw activations: {X_raw.shape}")

    # Build labels
    y = build_labels(all_ids, all_offsets, metadata)
    n_pos = y.sum()
    log.info(f"  Labels: {len(y)} residues, {n_pos} functional sites ({100*n_pos/len(y):.1f}%)")

    # IDENTICAL train/test split as 09 (same seed, same protein-level 80/20)
    rng = np.random.RandomState(42)
    n_proteins = len(all_ids)
    perm = rng.permutation(n_proteins)
    n_train = int(0.8 * n_proteins)
    train_prot_idx = set(perm[:n_train])

    train_residue_idx = []
    test_residue_idx = []
    for pi in range(n_proteins):
        start, L = all_offsets[pi]
        idx = list(range(start, start + L))
        if pi in train_prot_idx:
            train_residue_idx.extend(idx)
        else:
            test_residue_idx.extend(idx)

    train_residue_idx = np.array(train_residue_idx)
    test_residue_idx = np.array(test_residue_idx)

    X_train = X_raw[train_residue_idx]
    y_train = y[train_residue_idx]
    X_test = X_raw[test_residue_idx]
    y_test = y[test_residue_idx]

    log.info(f"  Train: {X_train.shape[0]} residues, {y_train.sum()} pos")
    log.info(f"  Test:  {X_test.shape[0]} residues, {y_test.sum()} pos")

    # Subsample for C search
    if X_train.shape[0] > MAX_CV_RESIDUES:
        cv_idx = rng.choice(X_train.shape[0], MAX_CV_RESIDUES, replace=False)
        X_cv, y_cv = X_train[cv_idx], y_train[cv_idx]
        log.info(f"  Subsampled {MAX_CV_RESIDUES} for C search")
    else:
        X_cv, y_cv = X_train, y_train

    n_val = int(0.2 * len(X_cv))
    val_idx = rng.choice(len(X_cv), n_val, replace=False)
    tr_mask = np.ones(len(X_cv), dtype=bool)
    tr_mask[val_idx] = False

    best_c, best_auroc = None, -1
    cv_results = {}
    for C in [0.0001, 0.0005, 0.001, 0.005]:
        log.info(f"    C={C}...")
        clf = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                                 max_iter=300, class_weight="balanced", random_state=42)
        clf.fit(X_cv[tr_mask], y_cv[tr_mask])
        val_proba = clf.decision_function(X_cv[val_idx])
        auroc = roc_auc_score(y_cv[val_idx], val_proba)
        log.info(f"      Val AUROC = {auroc:.4f}")
        cv_results[C] = auroc
        if auroc > best_auroc:
            best_auroc = auroc
            best_c = C

    log.info(f"  Best C={best_c} (val AUROC={best_auroc:.4f})")

    # Final model on capped training set
    if X_train.shape[0] > MAX_TRAIN_RESIDUES:
        final_idx = rng.choice(X_train.shape[0], MAX_TRAIN_RESIDUES, replace=False)
        X_final, y_final = X_train[final_idx], y_train[final_idx]
        log.info(f"  Training final model on {MAX_TRAIN_RESIDUES} residues")
    else:
        X_final, y_final = X_train, y_train

    clf = LogisticRegression(penalty="l1", C=best_c, solver="liblinear",
                             max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(X_final, y_final)
    test_proba = clf.decision_function(X_test)
    test_auroc = roc_auc_score(y_test, test_proba)
    report = classification_report(y_test, clf.predict(X_test), output_dict=True, zero_division=0)

    log.info(f"  Test AUROC: {test_auroc:.4f}")
    log.info(f"  Precision: {report.get('1', {}).get('precision', 0):.4f}")
    log.info(f"  Recall: {report.get('1', {}).get('recall', 0):.4f}")
    log.info(f"  Nonzero weights: {(np.abs(clf.coef_[0]) > 0).sum()}/{clf.coef_.shape[1]}")

    return {
        "model": model_name,
        "test_auroc": float(test_auroc),
        "best_C": float(best_c),
        "cv_results": {str(k): float(v) for k, v in cv_results.items()},
        "precision": float(report.get("1", {}).get("precision", 0)),
        "recall": float(report.get("1", {}).get("recall", 0)),
        "f1": float(report.get("1", {}).get("f1-score", 0)),
        "n_nonzero_weights": int((np.abs(clf.coef_[0]) > 0).sum()),
        "n_features": int(clf.coef_.shape[1]),
        "n_train": int(len(X_final)),
        "n_test": int(len(X_test)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    results = {}
    for model_name in ["esm3", "esm2"]:
        if not (ACT_ROOT / model_name / "residue_L33").exists():
            log.warning(f"No activations for {model_name}")
            continue
        results[model_name] = run_raw_probe(model_name, metadata)

    out_path = OUTPUT_DIR / "raw_baseline_probing.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved to {out_path}")

    # Comparison with SAE probes
    sae_path = OUTPUT_DIR / "interpretable_probing.json"
    if sae_path.exists():
        with open(sae_path) as f:
            sae_results = json.load(f)
        log.info(f"\n{'='*60}")
        log.info("SAE vs RAW COMPARISON")
        log.info(f"{'='*60}")
        for model in ["esm3", "esm2"]:
            if model in results and model in sae_results:
                log.info(f"  {model}: SAE AUROC={sae_results[model]['test_auroc']:.4f} "
                         f"vs Raw AUROC={results[model]['test_auroc']:.4f}")


if __name__ == "__main__":
    main()
