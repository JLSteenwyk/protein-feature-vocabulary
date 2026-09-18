#!/usr/bin/env python3
"""Analysis 1: Probe S vs S+St activations for secondary structure prediction.

Tests the dual encoding hypothesis: when structure tokens are absent, ESM-3
should encode predicted structure in the residual stream (higher SS probe accuracy).
When structure tokens are present, SS info is in the structure track, so probes
on the residual stream should perform worse.

Uses saved HDF5 activations at layers 16, 33, 42 under S-only and S+St conditions,
plus DSSP annotations for 199 proteins with AlphaFold structures.
"""

import json
import h5py
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = Path("/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/INTERPRETABILITY")
ACT_DIR = BASE / "data" / "activations" / "esm3_multimodal"
DSSP_PATH = BASE / "data" / "pilot" / "annotations" / "dssp_annotations.json"
RESULTS_DIR = BASE / "results" / "phase1" / "ss_probing"
FIG_DIR = BASE / "results" / "figures" / "phase1"

LAYERS = [16, 33, 42]
CONDITIONS = ["S", "S+St"]


def load_residue_index(condition: str) -> dict:
    """Load residue index mapping accession -> (start, end) in HDF5."""
    path = ACT_DIR / f"{condition}_residue_index.json"
    with open(path) as f:
        entries = json.load(f)
    return {e["accession"]: (e["start"], e["end"]) for e in entries}


def load_dssp() -> dict:
    """Load DSSP annotations. Returns {accession: [{ss3, rsa, ...}, ...]}"""
    with open(DSSP_PATH) as f:
        return json.load(f)


def build_dataset(layer: int, condition: str, dssp: dict, residue_index: dict):
    """Build X (activations) and y_ss, y_rsa from overlapping proteins.

    Returns X, y_ss (H=0, E=1, C=2), y_rsa (float), protein_ids
    """
    h5_path = ACT_DIR / f"{condition}_layer_{layer}.h5"

    # Find overlap: proteins in both residue_index and DSSP
    overlap = sorted(set(residue_index.keys()) & set(dssp.keys()))
    print(f"  [{condition} L{layer}] {len(overlap)} proteins with both activations and DSSP")

    ss_map = {"H": 0, "E": 1, "C": 2}

    X_parts = []
    y_ss_parts = []
    y_rsa_parts = []

    with h5py.File(h5_path, "r") as hf:
        act_data = hf["activations"]
        for acc in overlap:
            start, end = residue_index[acc]
            dssp_residues = dssp[acc]
            seq_len = end - start

            # DSSP and activation lengths must match
            if len(dssp_residues) != seq_len:
                # Try to align by taking min length
                n = min(len(dssp_residues), seq_len)
                if n == 0:
                    continue
                acts = act_data[start:start + n]
                residues = dssp_residues[:n]
            else:
                acts = act_data[start:end]
                residues = dssp_residues

            ss_labels = np.array([ss_map.get(r["ss3"], 2) for r in residues])
            rsa_values = np.array([r["rsa"] for r in residues])

            X_parts.append(acts[:])
            y_ss_parts.append(ss_labels)
            y_rsa_parts.append(rsa_values)

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y_ss = np.concatenate(y_ss_parts)
    y_rsa = np.concatenate(y_rsa_parts).astype(np.float32)

    print(f"  [{condition} L{layer}] {X.shape[0]} residues, dim={X.shape[1]}")
    print(f"  SS distribution: H={np.mean(y_ss==0):.1%}, E={np.mean(y_ss==1):.1%}, C={np.mean(y_ss==2):.1%}")

    return X, y_ss, y_rsa


def probe_ss(X, y_ss, n_splits=5):
    """Train SS classification probe with cross-validation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []

    for train_idx, test_idx in skf.split(X, y_ss):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(
            max_iter=500, C=1.0, solver='lbfgs',
            multi_class='multinomial', n_jobs=-1
        )
        clf.fit(X_train, y_ss[train_idx])

        y_pred = clf.predict(X_test)
        accs.append(accuracy_score(y_ss[test_idx], y_pred))
        f1s.append(f1_score(y_ss[test_idx], y_pred, average='macro'))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
    }


def probe_rsa(X, y_rsa, n_splits=5):
    """Train RSA regression probe with cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s, pearsons = [], []

    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        reg = Ridge(alpha=1.0)
        reg.fit(X_train, y_rsa[train_idx])

        y_pred = reg.predict(X_test)
        r2s.append(r2_score(y_rsa[test_idx], y_pred))
        pearsons.append(float(np.corrcoef(y_rsa[test_idx], y_pred)[0, 1]))

    return {
        "r2_mean": float(np.mean(r2s)),
        "r2_std": float(np.std(r2s)),
        "pearson_mean": float(np.mean(pearsons)),
        "pearson_std": float(np.std(pearsons)),
    }


def generate_figures(results: dict):
    """Generate publication figures for SS probing comparison."""
    layers = LAYERS

    # Extract metrics
    s_acc = [results[f"S_layer_{l}"]["ss"]["accuracy_mean"] for l in layers]
    sst_acc = [results[f"S+St_layer_{l}"]["ss"]["accuracy_mean"] for l in layers]
    s_acc_std = [results[f"S_layer_{l}"]["ss"]["accuracy_std"] for l in layers]
    sst_acc_std = [results[f"S+St_layer_{l}"]["ss"]["accuracy_std"] for l in layers]

    s_f1 = [results[f"S_layer_{l}"]["ss"]["f1_macro_mean"] for l in layers]
    sst_f1 = [results[f"S+St_layer_{l}"]["ss"]["f1_macro_mean"] for l in layers]
    s_f1_std = [results[f"S_layer_{l}"]["ss"]["f1_macro_std"] for l in layers]
    sst_f1_std = [results[f"S+St_layer_{l}"]["ss"]["f1_macro_std"] for l in layers]

    s_r2 = [results[f"S_layer_{l}"]["rsa"]["r2_mean"] for l in layers]
    sst_r2 = [results[f"S+St_layer_{l}"]["rsa"]["r2_mean"] for l in layers]
    s_r2_std = [results[f"S_layer_{l}"]["rsa"]["r2_std"] for l in layers]
    sst_r2_std = [results[f"S+St_layer_{l}"]["rsa"]["r2_std"] for l in layers]

    s_pearson = [results[f"S_layer_{l}"]["rsa"]["pearson_mean"] for l in layers]
    sst_pearson = [results[f"S+St_layer_{l}"]["rsa"]["pearson_mean"] for l in layers]

    # Figure 8: SS Probing Comparison (2-panel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(layers))
    width = 0.35

    # Panel A: SS Accuracy
    ax = axes[0]
    bars1 = ax.bar(x - width/2, s_acc, width, yerr=s_acc_std,
                   label='S-only', color='#2196F3', capsize=4, alpha=0.85)
    bars2 = ax.bar(x + width/2, sst_acc, width, yerr=sst_acc_std,
                   label='S+St', color='#FF5722', capsize=4, alpha=0.85)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('SS3 Accuracy', fontsize=12)
    ax.set_title('A. Secondary Structure Classification', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Panel B: RSA Regression
    ax = axes[1]
    bars1 = ax.bar(x - width/2, s_pearson, width,
                   label='S-only', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, sst_pearson, width,
                   label='S+St', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Pearson r (RSA prediction)', fontsize=12)
    ax.set_title('B. Solvent Accessibility Regression', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f'{h:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig8_ss_probing_comparison.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 9: Dual encoding summary (delta plot)
    fig, ax = plt.subplots(figsize=(8, 5))

    delta_acc = [s - sst for s, sst in zip(s_acc, sst_acc)]
    delta_f1 = [s - sst for s, sst in zip(s_f1, sst_f1)]
    delta_r2 = [s - sst for s, sst in zip(s_r2, sst_r2)]

    x = np.arange(len(layers))
    width = 0.25

    ax.bar(x - width, delta_acc, width, label='SS Accuracy (S - S+St)', color='#4CAF50', alpha=0.85)
    ax.bar(x, delta_f1, width, label='SS F1 macro (S - S+St)', color='#2196F3', alpha=0.85)
    ax.bar(x + width, delta_r2, width, label='RSA R² (S - S+St)', color='#FF9800', alpha=0.85)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Probe Performance Difference (S-only minus S+St)', fontsize=12)
    ax.set_title('Dual Encoding: S-only Encodes More Structure\n(Positive = S-only is better)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(fontsize=10)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig9_dual_encoding_delta.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved to {FIG_DIR}/fig8_*, fig9_*")


def main():
    print("=" * 70)
    print("Analysis 1: SS Probing Comparison (S-only vs S+St)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load DSSP annotations
    print("\nLoading DSSP annotations...")
    dssp = load_dssp()
    print(f"  {len(dssp)} proteins with DSSP data")

    # Load residue indices
    s_index = load_residue_index("S")
    sst_index = load_residue_index("S+St")

    # Find common proteins across both conditions and DSSP
    common = sorted(set(s_index.keys()) & set(sst_index.keys()) & set(dssp.keys()))
    print(f"  {len(common)} proteins common to S, S+St, and DSSP")

    results = {"n_common_proteins": len(common)}

    for layer in LAYERS:
        for condition in CONDITIONS:
            print(f"\n--- {condition} Layer {layer} ---")

            idx = s_index if condition == "S" else sst_index
            # Use only common proteins for fair comparison
            common_idx = {acc: idx[acc] for acc in common if acc in idx}

            X, y_ss, y_rsa = build_dataset(layer, condition, dssp, common_idx)

            print(f"  Training SS probe (5-fold CV)...")
            ss_results = probe_ss(X, y_ss)
            print(f"  SS Accuracy: {ss_results['accuracy_mean']:.4f} +/- {ss_results['accuracy_std']:.4f}")
            print(f"  SS F1 macro: {ss_results['f1_macro_mean']:.4f} +/- {ss_results['f1_macro_std']:.4f}")

            print(f"  Training RSA probe (5-fold CV)...")
            rsa_results = probe_rsa(X, y_rsa)
            print(f"  RSA R²:      {rsa_results['r2_mean']:.4f} +/- {rsa_results['r2_std']:.4f}")
            print(f"  RSA Pearson:  {rsa_results['pearson_mean']:.4f} +/- {rsa_results['pearson_std']:.4f}")

            key = f"{condition}_layer_{layer}"
            results[key] = {
                "ss": ss_results,
                "rsa": rsa_results,
                "n_residues": int(X.shape[0]),
            }

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Dual Encoding Hypothesis Test")
    print("=" * 70)
    print(f"{'Layer':>6} | {'SS Acc (S)':>12} | {'SS Acc (S+St)':>14} | {'Delta':>8} | {'RSA r (S)':>10} | {'RSA r (S+St)':>13} | {'Delta':>8}")
    print("-" * 90)

    for layer in LAYERS:
        s_key = f"S_layer_{layer}"
        sst_key = f"S+St_layer_{layer}"
        s_acc = results[s_key]["ss"]["accuracy_mean"]
        sst_acc = results[sst_key]["ss"]["accuracy_mean"]
        s_r = results[s_key]["rsa"]["pearson_mean"]
        sst_r = results[sst_key]["rsa"]["pearson_mean"]

        d_acc = s_acc - sst_acc
        d_r = s_r - sst_r

        marker_acc = "***" if abs(d_acc) > 0.02 else "**" if abs(d_acc) > 0.01 else "*" if abs(d_acc) > 0.005 else ""
        marker_r = "***" if abs(d_r) > 0.05 else "**" if abs(d_r) > 0.02 else "*" if abs(d_r) > 0.01 else ""

        print(f"{layer:>6} | {s_acc:>12.4f} | {sst_acc:>14.4f} | {d_acc:>+7.4f}{marker_acc} | {s_r:>10.4f} | {sst_r:>13.4f} | {d_r:>+7.4f}{marker_r}")

    # Save results
    out_path = RESULTS_DIR / "ss_probing_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
