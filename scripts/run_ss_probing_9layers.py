#!/usr/bin/env python3
"""SS/RSA probing across all 9 S-only ESM-3 layers.

Probes secondary structure (3-class: H/E/C) and relative solvent accessibility
at layers 0, 6, 12, 18, 24, 30, 36, 42, 47 using S-only activations for
the 199 proteins that have both activations and DSSP annotations.

This reveals *when* the model internally predicts structure from sequence alone.
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
ACT_DIR = BASE / "data" / "activations" / "esm3"
DSSP_PATH = BASE / "data" / "pilot" / "annotations" / "dssp_annotations.json"
RESULTS_DIR = BASE / "results" / "phase1" / "ss_probing"
FIG_DIR = BASE / "results" / "figures" / "phase1"

LAYERS = [0, 6, 12, 18, 24, 30, 36, 42, 47]
SS_MAP = {"H": 0, "E": 1, "C": 2}


def load_data():
    """Load residue index and DSSP annotations, find overlap."""
    with open(ACT_DIR / "residue_index.json") as f:
        res_idx = json.load(f)
    idx_map = {e["accession"]: (e["start"], e["end"]) for e in res_idx}

    with open(DSSP_PATH) as f:
        dssp = json.load(f)

    overlap = sorted(set(idx_map.keys()) & set(dssp.keys()))
    print(f"  {len(idx_map)} proteins with activations")
    print(f"  {len(dssp)} proteins with DSSP")
    print(f"  {len(overlap)} in overlap")

    return idx_map, dssp, overlap


def build_labels(dssp, idx_map, overlap):
    """Build SS and RSA label arrays for overlapping proteins."""
    y_ss_parts = []
    y_rsa_parts = []
    slices = []  # (start, end) in HDF5 for each protein

    for acc in overlap:
        start, end = idx_map[acc]
        residues = dssp[acc]
        seq_len = end - start
        n = min(len(residues), seq_len)
        if n == 0:
            continue

        ss_labels = np.array([SS_MAP.get(r["ss3"], 2) for r in residues[:n]])
        rsa_values = np.array([r["rsa"] for r in residues[:n]])

        y_ss_parts.append(ss_labels)
        y_rsa_parts.append(rsa_values)
        slices.append((start, start + n))

    y_ss = np.concatenate(y_ss_parts)
    y_rsa = np.concatenate(y_rsa_parts).astype(np.float32)

    print(f"  {len(slices)} proteins, {y_ss.shape[0]} residues")
    print(f"  SS: H={np.mean(y_ss==0):.1%}, E={np.mean(y_ss==1):.1%}, C={np.mean(y_ss==2):.1%}")
    return y_ss, y_rsa, slices


def load_activations(layer, slices):
    """Load activations for the overlapping residues at a given layer."""
    h5_path = ACT_DIR / f"layer_{layer}.h5"
    parts = []
    with h5py.File(h5_path, "r") as hf:
        act = hf["activations"]
        for start, end in slices:
            parts.append(act[start:end][:])
    return np.concatenate(parts, axis=0).astype(np.float32)


def probe_ss(X, y_ss, n_splits=5):
    """SS classification with 5-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs, f1s = [], []
    for train_idx, test_idx in skf.split(X, y_ss):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs',
                                 multi_class='multinomial', n_jobs=-1)
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
    """RSA regression with 5-fold CV."""
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


def generate_figures(results):
    """Generate layer-wise SS/RSA probing figure."""
    layers = LAYERS
    ss_acc = [results[f"layer_{l}"]["ss"]["accuracy_mean"] for l in layers]
    ss_acc_std = [results[f"layer_{l}"]["ss"]["accuracy_std"] for l in layers]
    ss_f1 = [results[f"layer_{l}"]["ss"]["f1_macro_mean"] for l in layers]
    ss_f1_std = [results[f"layer_{l}"]["ss"]["f1_macro_std"] for l in layers]
    rsa_r = [results[f"layer_{l}"]["rsa"]["pearson_mean"] for l in layers]
    rsa_r_std = [results[f"layer_{l}"]["rsa"]["pearson_std"] for l in layers]
    rsa_r2 = [results[f"layer_{l}"]["rsa"]["r2_mean"] for l in layers]
    rsa_r2_std = [results[f"layer_{l}"]["rsa"]["r2_std"] for l in layers]

    # Also load the S+St results at layers 16, 33, 42 for comparison
    sst_path = RESULTS_DIR / "ss_probing_results.json"
    sst_results = None
    if sst_path.exists():
        with open(sst_path) as f:
            sst_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: SS3 accuracy across layers
    ax = axes[0]
    ax.errorbar(layers, ss_acc, yerr=ss_acc_std, marker='o', linewidth=2,
                color='#2196F3', label='S-only SS Accuracy', capsize=4, markersize=7)
    ax.errorbar(layers, ss_f1, yerr=ss_f1_std, marker='s', linewidth=2,
                color='#4CAF50', label='S-only SS F1 (macro)', capsize=4, markersize=6)

    # Add S+St reference points if available
    if sst_results:
        sst_layers = [16, 33, 42]
        sst_acc_vals = [sst_results[f"S+St_layer_{l}"]["ss"]["accuracy_mean"] for l in sst_layers]
        sst_f1_vals = [sst_results[f"S+St_layer_{l}"]["ss"]["f1_macro_mean"] for l in sst_layers]
        ax.scatter(sst_layers, sst_acc_vals, marker='D', s=80, color='#FF5722',
                   zorder=5, label='S+St SS Accuracy (ref)')
        ax.scatter(sst_layers, sst_f1_vals, marker='D', s=60, color='#FF9800',
                   zorder=5, label='S+St SS F1 (ref)')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('A. Secondary Structure Prediction\nAcross ESM-3 Layers (S-only)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.3, 1.0)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)

    # Annotate key transitions
    max_gain_idx = np.argmax(np.diff(ss_acc))
    ax.annotate(f'Largest gain:\nL{layers[max_gain_idx]}→L{layers[max_gain_idx+1]}',
                xy=(layers[max_gain_idx+1], ss_acc[max_gain_idx+1]),
                xytext=(layers[max_gain_idx+1]+3, ss_acc[max_gain_idx+1]-0.08),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray')

    # Panel B: RSA prediction across layers
    ax = axes[1]
    ax.errorbar(layers, rsa_r, yerr=rsa_r_std, marker='o', linewidth=2,
                color='#2196F3', label='S-only Pearson r', capsize=4, markersize=7)
    ax.errorbar(layers, rsa_r2, yerr=rsa_r2_std, marker='s', linewidth=2,
                color='#9C27B0', label='S-only R²', capsize=4, markersize=6)

    if sst_results:
        sst_pearson = [sst_results[f"S+St_layer_{l}"]["rsa"]["pearson_mean"] for l in sst_layers]
        sst_r2 = [sst_results[f"S+St_layer_{l}"]["rsa"]["r2_mean"] for l in sst_layers]
        ax.scatter(sst_layers, sst_pearson, marker='D', s=80, color='#FF5722',
                   zorder=5, label='S+St Pearson r (ref)')
        ax.scatter(sst_layers, sst_r2, marker='D', s=60, color='#FF9800',
                   zorder=5, label='S+St R² (ref)')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('B. Solvent Accessibility Prediction\nAcross ESM-3 Layers (S-only)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig13_ss_probing_9layers.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    # Figure 14: Combined S-only trajectory + S+St reference — clean version
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(layers,
                    [a - s for a, s in zip(ss_acc, ss_acc_std)],
                    [a + s for a, s in zip(ss_acc, ss_acc_std)],
                    alpha=0.15, color='#2196F3')
    ax.plot(layers, ss_acc, 'o-', linewidth=2.5, color='#2196F3',
            label='SS3 Accuracy (S-only)', markersize=8)

    ax.fill_between(layers,
                    [r - s for r, s in zip(rsa_r, rsa_r_std)],
                    [r + s for r, s in zip(rsa_r, rsa_r_std)],
                    alpha=0.15, color='#FF9800')
    ax.plot(layers, rsa_r, 's-', linewidth=2.5, color='#FF9800',
            label='RSA Pearson r (S-only)', markersize=7)

    if sst_results:
        # S+St as horizontal bands (nearly constant across layers)
        sst_ss_mean = np.mean(sst_acc_vals)
        sst_rsa_mean = np.mean(sst_pearson)
        ax.axhline(y=sst_ss_mean, color='#2196F3', linestyle='--', alpha=0.5,
                   linewidth=1.5, label=f'SS3 Acc (S+St avg={sst_ss_mean:.3f})')
        ax.axhline(y=sst_rsa_mean, color='#FF9800', linestyle='--', alpha=0.5,
                   linewidth=1.5, label=f'RSA r (S+St avg={sst_rsa_mean:.3f})')

    ax.set_xlabel('ESM-3 Layer', fontsize=13)
    ax.set_ylabel('Probe Performance', fontsize=13)
    ax.set_title('Structural Information Emergence in ESM-3\n(Sequence-Only vs Structure-Provided)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-1, 48)
    ax.set_ylim(0.2, 1.0)
    ax.set_xticks(layers)
    ax.grid(alpha=0.3)

    # Annotate regions
    ax.axvspan(-0.5, 9.5, alpha=0.05, color='red', label='_nolegend_')
    ax.axvspan(27.5, 40.5, alpha=0.05, color='blue', label='_nolegend_')
    ax.text(4.5, 0.25, 'Attention\nIntegration\nZone', ha='center', fontsize=9,
            color='red', alpha=0.6, style='italic')
    ax.text(34, 0.25, 'CKA\nConvergence\nZone', ha='center', fontsize=9,
            color='blue', alpha=0.6, style='italic')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig14_structural_emergence.{ext}', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Figures saved: fig13_ss_probing_9layers, fig14_structural_emergence")


def main():
    print("=" * 70)
    print("SS/RSA Probing Across All 9 S-only ESM-3 Layers")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    idx_map, dssp, overlap = load_data()

    print("\nBuilding labels...")
    y_ss, y_rsa, slices = build_labels(dssp, idx_map, overlap)

    results = {
        "description": "SS3/RSA probing across 9 S-only ESM-3 layers",
        "n_proteins": len(slices),
        "n_residues": int(y_ss.shape[0]),
        "layers": LAYERS,
    }

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        X = load_activations(layer, slices)
        print(f"  Activations: {X.shape}")

        print(f"  SS probe (5-fold CV)...")
        ss_res = probe_ss(X, y_ss)
        print(f"  Accuracy: {ss_res['accuracy_mean']:.4f} ± {ss_res['accuracy_std']:.4f}")
        print(f"  F1 macro: {ss_res['f1_macro_mean']:.4f} ± {ss_res['f1_macro_std']:.4f}")

        print(f"  RSA probe (5-fold CV)...")
        rsa_res = probe_rsa(X, y_rsa)
        print(f"  Pearson r: {rsa_res['pearson_mean']:.4f} ± {rsa_res['pearson_std']:.4f}")
        print(f"  R²:        {rsa_res['r2_mean']:.4f} ± {rsa_res['r2_std']:.4f}")

        results[f"layer_{layer}"] = {"ss": ss_res, "rsa": rsa_res}

        del X  # free memory

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Structural Information Emergence (S-only)")
    print("=" * 70)
    print(f"{'Layer':>6} | {'SS Acc':>10} | {'SS F1':>10} | {'RSA r':>10} | {'RSA R²':>10}")
    print("-" * 58)
    for layer in LAYERS:
        r = results[f"layer_{layer}"]
        print(f"{layer:>6} | {r['ss']['accuracy_mean']:>10.4f} | {r['ss']['f1_macro_mean']:>10.4f} | "
              f"{r['rsa']['pearson_mean']:>10.4f} | {r['rsa']['r2_mean']:>10.4f}")

    # Identify steepest gains
    accs = [results[f"layer_{l}"]["ss"]["accuracy_mean"] for l in LAYERS]
    gains = np.diff(accs)
    max_gain_idx = np.argmax(gains)
    print(f"\nSteepest SS gain: Layer {LAYERS[max_gain_idx]} → {LAYERS[max_gain_idx+1]} "
          f"(+{gains[max_gain_idx]:.4f})")

    rsas = [results[f"layer_{l}"]["rsa"]["pearson_mean"] for l in LAYERS]
    rsa_gains = np.diff(rsas)
    max_rsa_idx = np.argmax(rsa_gains)
    print(f"Steepest RSA gain: Layer {LAYERS[max_rsa_idx]} → {LAYERS[max_rsa_idx+1]} "
          f"(+{rsa_gains[max_rsa_idx]:.4f})")

    # Save
    out_path = RESULTS_DIR / "ss_probing_9layers.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
