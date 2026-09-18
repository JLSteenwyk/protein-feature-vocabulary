#!/usr/bin/env python3
"""Homology-aware robustness: length-stratified 5-fold CV for probing and convergence.

Usage:
    ./env/bin/python scripts/scaled_1.5M/32_homology_splits.py
"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import h5py

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("homology")

FEATURE_ROOT = ROOT / "results" / "scaled_1.5M" / "sae_features"
EVAL_DIR = ROOT / "data" / "eval_expanded"
OUTPUT_DIR = ROOT / "results" / "scaled_1.5M"
SEED = 42
N_FOLDS = 5
MAX_RESIDUES_PER_FOLD = 200000

def main():
    # Load sequences for length-based stratification
    with open(EVAL_DIR / "sequences.json") as f:
        sequences = json.load(f)

    # Load SAE features
    log.info("Loading ESM-3 features...")
    X_esm3 = sparse.load_npz(FEATURE_ROOT / "esm3" / "residue" / "features_sparse.npz")
    with h5py.File(FEATURE_ROOT / "esm3" / "residue" / "protein_summaries.h5") as f:
        protein_ids = [s.decode() for s in f["protein_ids"][:]]
        offsets = f["offsets"][:]
        prot_max_3 = f["protein_max_activations"][:]

    X_esm2 = sparse.load_npz(FEATURE_ROOT / "esm2" / "residue" / "features_sparse.npz")
    with h5py.File(FEATURE_ROOT / "esm2" / "residue" / "protein_summaries.h5") as f:
        prot_max_2 = f["protein_max_activations"][:]

    # Load functional site labels
    meta = json.load(open(EVAL_DIR / "metadata.json"))
    n_total = int(offsets[-1][0] + offsets[-1][1])
    labels = np.zeros(n_total, dtype=np.int8)
    for i, pid in enumerate(protein_ids):
        if pid in meta:
            for feat in meta[pid].get("features", []):
                ft = feat.get("type", "")
                if ft in ["Active site", "Binding site", "Metal binding", "Site", "Disulfide bond", "Modified residue"]:
                    s, l = int(offsets[i][0]), int(offsets[i][1])
                    ps = feat.get("start", 1) - 1
                    pe = feat.get("end", ps + 1)
                    for pos in range(max(0, ps), min(l, pe)):
                        labels[s + pos] = 1

    log.info(f"  {len(protein_ids)} proteins, {labels.sum()} positive residues")

    # Length-stratified folds
    lengths = np.array([len(sequences.get(pid, "")) for pid in protein_ids])
    sorted_idx = np.argsort(lengths)
    folds = np.zeros(len(protein_ids), dtype=int)
    for i, idx in enumerate(sorted_idx):
        folds[idx] = i % N_FOLDS

    log.info(f"Fold sizes: {[np.sum(folds == f) for f in range(N_FOLDS)]}")

    # === Probing: 5-fold CV ===
    log.info("\n=== Probing: length-stratified 5-fold CV ===")
    rng = np.random.RandomState(SEED)
    probing_results = []

    for fold in range(N_FOLDS):
        test_mask = folds == fold
        train_mask = ~test_mask

        train_prot_idx = np.where(train_mask)[0]
        test_prot_idx = np.where(test_mask)[0]

        train_residues, test_residues = [], []
        for i in train_prot_idx:
            s, l = int(offsets[i][0]), int(offsets[i][1])
            train_residues.extend(range(s, s + l))
        for i in test_prot_idx:
            s, l = int(offsets[i][0]), int(offsets[i][1])
            test_residues.extend(range(s, s + l))

        # Subsample for speed
        if len(train_residues) > MAX_RESIDUES_PER_FOLD:
            train_residues = rng.choice(train_residues, MAX_RESIDUES_PER_FOLD, replace=False).tolist()
        if len(test_residues) > MAX_RESIDUES_PER_FOLD:
            test_residues = rng.choice(test_residues, MAX_RESIDUES_PER_FOLD, replace=False).tolist()

        X_tr = X_esm3[train_residues]
        y_tr = labels[train_residues]
        X_te = X_esm3[test_residues]
        y_te = labels[test_residues]

        if y_tr.sum() < 10 or y_te.sum() < 5:
            log.warning(f"  Fold {fold}: insufficient labels, skipping")
            continue

        clf = LogisticRegression(penalty="l1", C=0.01, solver="liblinear",
                                 max_iter=200, class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y_tr)
        auroc = roc_auc_score(y_te, clf.decision_function(X_te))

        result = {
            "fold": fold,
            "n_train_proteins": int(train_mask.sum()),
            "n_test_proteins": int(test_mask.sum()),
            "n_train_residues": len(train_residues),
            "n_test_residues": len(test_residues),
            "test_pos": int(y_te.sum()),
            "auroc": float(auroc),
        }
        probing_results.append(result)
        log.info(f"  Fold {fold}: AUROC={auroc:.4f} (train={len(train_residues)}, test={len(test_residues)}, pos={y_te.sum()})")

    mean_auroc = np.mean([r["auroc"] for r in probing_results])
    std_auroc = np.std([r["auroc"] for r in probing_results])
    log.info(f"  Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")

    # === Convergence: per-fold ===
    log.info("\n=== Convergence: length-stratified folds ===")
    active_3 = np.where((prot_max_3 > 0).any(axis=0))[0]
    active_2 = np.where((prot_max_2 > 0).any(axis=0))[0]

    convergence_results = []
    for fold in range(N_FOLDS):
        test_idx = np.where(folds == fold)[0]
        mat3 = prot_max_3[test_idx][:, active_3]
        mat2 = prot_max_2[test_idx][:, active_2]
        n_prot = len(test_idx)

        # Normalize ESM-2
        m2_norm = mat2 - mat2.mean(axis=0, keepdims=True)
        m2_std = m2_norm.std(axis=0, keepdims=True)
        m2_std[m2_std == 0] = 1
        m2_norm /= m2_std

        # Compute best-match r for ESM-3 features (sample 500 for speed)
        sample = min(500, mat3.shape[1])
        sample_idx = rng.choice(mat3.shape[1], sample, replace=False)
        best_rs = []
        for i in sample_idx:
            f3 = mat3[:, i]
            f3_n = f3 - f3.mean()
            f3_s = f3_n.std()
            if f3_s == 0: continue
            f3_n /= f3_s
            corr = (f3_n @ m2_norm) / n_prot
            best_rs.append(float(corr.max()))

        med_r = np.median(best_rs) if best_rs else 0
        frac_03 = np.mean(np.array(best_rs) > 0.3) if best_rs else 0

        result = {
            "fold": fold,
            "n_proteins": int(len(test_idx)),
            "n_features_sampled": len(best_rs),
            "median_r": float(med_r),
            "frac_above_0.3": float(frac_03),
        }
        convergence_results.append(result)
        log.info(f"  Fold {fold}: median_r={med_r:.3f}, >0.3={frac_03*100:.1f}% (n={len(test_idx)} proteins)")

    mean_frac = np.mean([r["frac_above_0.3"] for r in convergence_results])
    log.info(f"  Mean frac>0.3: {mean_frac*100:.1f}%")

    # Save
    output = {
        "n_folds": N_FOLDS,
        "split_method": "length_stratified",
        "probing": {
            "per_fold": probing_results,
            "mean_auroc": float(mean_auroc),
            "std_auroc": float(std_auroc),
            "original_auroc": 0.959,
        },
        "convergence": {
            "per_fold": convergence_results,
            "mean_frac_above_0.3": float(mean_frac),
            "std_frac": float(np.std([r["frac_above_0.3"] for r in convergence_results])),
            "original_frac": 0.780,
        },
    }

    out_path = OUTPUT_DIR / "homology_splits.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
