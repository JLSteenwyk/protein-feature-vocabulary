#!/usr/bin/env python3
"""Generate InterPLM-style SAE feature visualization figures.

For selected SAE features, creates panels showing:
  - Per-residue activation bar chart with amino acid letters at peaks
  - 3D protein structure colored by feature activation intensity (cyan→pink)

Inspired by Fig 1c,d in InterPLM (Nature Methods, 2025).

Usage:
    ./env/bin/python scripts/unified/generate_sae_feature_figures.py \
        --model esm2 --layer 24 --device cpu --n-candidates 100
    ./env/bin/python scripts/unified/generate_sae_feature_figures.py \
        --model esm3 --layer 33 --condition S --device cpu --n-candidates 100
"""

import sys
import os
import json
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBParser

from sae.model import TopKSAE, SAEConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sae_fig")


# ── Color scheme (InterPLM-style) ─────────────────────────────────────
CYAN = np.array([0.0, 0.737, 0.831])     # #00BCD4
PINK = np.array([0.906, 0.161, 0.541])   # #E7298A
MAGENTA = "#C51B8A"
LIGHT_CYAN = "#B2EBF2"


def load_sae(model_name, layer, condition="S"):
    """Load a trained TopK SAE checkpoint.

    For ESM-3, prefers the non-scaled SAE (which has matching annotations)
    but falls back to scaled if not available.
    """
    if model_name == "esm2":
        ckpt_dir = ROOT / "models" / "sae" / "esm2_scaled" / f"layer_{layer}_topk"
    else:
        # Prefer non-scaled (has matching annotations from phase1)
        ckpt_dir = ROOT / "models" / "sae" / "esm3" / f"{condition}_layer_{layer}_topk"
        if not ckpt_dir.exists():
            ckpt_dir = ROOT / "models" / "sae" / "esm3_scaled" / f"{condition}_layer_{layer}_topk"

    ckpt_path = ckpt_dir / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "final.pt"
    log.info(f"Loading SAE from {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["sae_config"]
    sae = TopKSAE(config)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    return sae, config


def build_protein_offsets(sequences):
    """Build mapping from accession to (start_idx, length) in the concatenated HDF5."""
    accs = sorted(sequences.keys())
    offsets = {}
    idx = 0
    for acc in accs:
        L = len(sequences[acc])
        offsets[acc] = (idx, L)
        idx += L
    return offsets, accs


def parse_pdb_ca(pdb_path):
    """Parse PDB file, return Cα coordinates as (N, 3) array."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", str(pdb_path))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
        break
    return np.array(coords)


def batch_encode_feature(sae, h5_path, feature_indices, batch_size=4096):
    """Batch-encode the entire HDF5 and extract activations for specific features.

    Returns dict: feature_idx -> (total_residues,) array of activations.
    Much faster than per-protein encoding.
    """
    with h5py.File(h5_path, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]

        result = {fi: np.zeros(total, dtype=np.float32) for fi in feature_indices}

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            raw = torch.tensor(acts[start:end], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw)  # (batch, dict_size)

            for fi in feature_indices:
                result[fi][start:end] = z[:, fi].numpy()

            if (start // batch_size) % 20 == 0:
                log.info(f"  Encoded {end}/{total} residues...")

    return result


def find_top_protein_from_precomputed(feat_acts_full, offsets, accs,
                                       sequences, structure_dir, max_length=400):
    """Given pre-computed per-residue activations, find the best protein."""
    best_acc = None
    best_max = -1.0
    best_start = 0
    best_L = 0

    for acc in accs:
        start, L = offsets[acc]
        seq = sequences[acc]
        L = min(len(seq), L)
        if L > max_length:
            continue
        if start + L > len(feat_acts_full):
            continue

        local_max = float(feat_acts_full[start:start+L].max())
        if local_max > best_max:
            if structure_dir and (structure_dir / f"{acc}.pdb").exists():
                best_max = local_max
                best_acc = acc
                best_start = start
                best_L = L

    if best_acc is None:
        return None, None, -1.0

    return best_acc, feat_acts_full[best_start:best_start+best_L].copy(), best_max


def select_features_from_annotations(model_name, layer, condition="S"):
    """Select interesting features using existing annotation files."""
    selected = []

    if model_name == "esm2":
        ann_path = ROOT / "results" / "phase0" / "sae_reproduction" / "feature_annotations.json"
        struct_path = None
    else:
        ann_path = ROOT / "results" / "phase1" / "modality_saes" / f"{condition}_layer_{layer}_annotations.json"
        struct_path = ROOT / "results" / "phase1" / "modality_saes" / "structural_annotations" / f"{condition}_layer_{layer}_structural.json"

    # Load functional annotations
    func_ann = {}
    if ann_path.exists():
        with open(ann_path) as f:
            for a in json.load(f):
                func_ann[a["feature_idx"]] = a

    # Load structural annotations
    struct_ann = {}
    if struct_path and struct_path.exists():
        with open(struct_path) as f:
            data = json.load(f)
        features = data if isinstance(data, list) else data.get("features", [])
        for a in features:
            struct_ann[a["feature_idx"]] = a

    # Category 1: AA-specific features (2 picks — different AAs)
    aa_feats = []
    for fi, a in func_ann.items():
        top_aa = a.get("top_enriched_aa", [])
        if top_aa and top_aa[0][1] > 5.0 and a["activation_freq"] < 0.15:
            aa_feats.append((fi, top_aa[0][0], top_aa[0][1], a["activation_freq"]))
    aa_feats.sort(key=lambda x: -x[2])

    seen_aa = set()
    for fi, aa, enr, freq in aa_feats:
        if aa not in seen_aa and len(selected) < 2:
            label = f"Activates on {aa} residues ({enr:.0f}x enrichment)"
            selected.append((fi, "aa_specific", label))
            seen_aa.add(aa)

    # Category 2: Functional site features
    func_feats = [(fi, a["functional_site_enrichment"], a["functional_site_frac"])
                  for fi, a in func_ann.items()
                  if a["functional_site_frac"] > 0.04 and a["functional_site_enrichment"] > 2.0]
    func_feats.sort(key=lambda x: -x[1])

    for fi, enr, frac in func_feats[:1]:
        label = f"Activates on functional sites ({frac:.0%} at annotated sites)"
        selected.append((fi, "functional", label))

    # Category 3: SS-specific features (helix + sheet)
    if struct_ann:
        for target_ss, ss_name in [("H", "alpha helices"), ("E", "beta strands")]:
            ss_feats = []
            for fi, a in struct_ann.items():
                ss_enr = a.get("ss_enrichment", {})
                if ss_enr.get(target_ss, 0) > 1.5:
                    ss_feats.append((fi, ss_enr[target_ss], a.get("rsa_correlation", 0)))
            ss_feats.sort(key=lambda x: -x[1])
            if ss_feats and len(selected) < 5:
                fi, enr, rsa = ss_feats[0]
                label = f"Activates on {ss_name} (SS enrichment {enr:.1f}x)"
                selected.append((fi, "ss_specific", label))

    # Category 4: Surface vs buried
    if struct_ann:
        buried = [(fi, a["rsa_correlation"]) for fi, a in struct_ann.items()
                  if a.get("surface_preference") == "buried" and abs(a["rsa_correlation"]) > 0.15]
        buried.sort(key=lambda x: x[1])  # Most negative RSA correlation = most buried
        if buried and len(selected) < 6:
            fi, rsa = buried[0]
            label = f"Activates on buried residues (RSA corr={rsa:.2f})"
            selected.append((fi, "structural", label))

    # Fill remaining with selective features (low freq, high enrichment)
    if len(selected) < 6:
        selective = [(fi, a["activation_freq"], a.get("top_enriched_aa", [[None, 0]])[0])
                     for fi, a in func_ann.items()
                     if 0.01 < a["activation_freq"] < 0.05
                     and fi not in [s[0] for s in selected]]
        selective.sort(key=lambda x: x[1])
        for fi, freq, top in selective[:6-len(selected)]:
            aa = top[0] if top else "?"
            label = f"Selective feature (active in {freq:.1%} of residues)"
            selected.append((fi, "selective", label))

    return selected


def plot_feature_panel(ax_bar, ax_3d, seq, feat_acts, ca_coords, feature_idx,
                       label="", metadata=None):
    """Draw one feature panel: bar chart (left) + 3D structure (right)."""
    L = len(feat_acts)

    # Normalize activations to [0, 1]
    max_act = feat_acts.max()
    if max_act > 0:
        norm_acts = feat_acts / max_act
    else:
        norm_acts = feat_acts.copy()

    # ── Bar chart ──
    positions = np.arange(L)

    # Color bars by activation intensity (cyan → pink)
    bar_colors = np.zeros((L, 3))
    for i in range(L):
        a = norm_acts[i]
        bar_colors[i] = (1 - a) * CYAN + a * PINK

    ax_bar.bar(positions, norm_acts, width=1.0, color=bar_colors, edgecolor="none")

    # Add amino acid letters at top of high-activation bars
    threshold = 0.5
    peak_positions = np.where(norm_acts > threshold)[0]

    # Keep only top 20 peaks for readability
    if len(peak_positions) > 20:
        sorted_peaks = peak_positions[np.argsort(-norm_acts[peak_positions])]
        peak_positions = sorted(sorted_peaks[:20])

    for pos in peak_positions:
        if pos < len(seq):
            ax_bar.text(pos, norm_acts[pos] + 0.02, seq[pos],
                       ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                       color=MAGENTA)

    ax_bar.set_xlim(-1, L + 1)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_xlabel("Sequence position", fontsize=9)
    ax_bar.set_ylabel("Feature activation", fontsize=9)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(labelsize=7)

    if label:
        ax_bar.set_title(label, fontsize=9, color=MAGENTA, fontweight="bold", loc="left")

    # Annotate functional sites from metadata
    if metadata:
        features = metadata.get("features", [])
        for feat in features:
            ftype = feat.get("type", "")
            if ftype in ("Active site", "Binding site", "Disulfide bond"):
                start = feat.get("start", 0) - 1  # to 0-indexed
                end = feat.get("end", start + 1)
                if 0 <= start < L:
                    ax_bar.axvspan(start - 0.5, min(end, L) - 0.5,
                                  alpha=0.08, color="gray", zorder=0)
                    # Label if not too crowded
                    mid = (start + min(end, L)) / 2
                    desc = feat.get("description", ftype)
                    if len(desc) > 20:
                        desc = ftype
                    ax_bar.text(mid, -0.08, desc, ha="center", va="top",
                               fontsize=5.5, color="gray", alpha=0.8, style="italic",
                               rotation=0, clip_on=True)

    # ── 3D structure ──
    n_coords = min(len(ca_coords), L)
    if n_coords < 3:
        ax_3d.text(0.5, 0.5, 0.5, "No structure", ha="center", transform=ax_3d.transAxes)
        return

    coords = ca_coords[:n_coords]
    acts_3d = norm_acts[:n_coords]

    # Color: cyan → pink by activation
    rgba = np.zeros((n_coords, 4))
    for i in range(n_coords):
        a = acts_3d[i]
        rgba[i, :3] = (1 - a) * CYAN + a * PINK
        rgba[i, 3] = 0.5 + 0.5 * a  # More opaque for high activation

    # Plot backbone as thin line
    ax_3d.plot(coords[:, 0], coords[:, 1], coords[:, 2],
              color=(*CYAN, 0.25), linewidth=0.4, zorder=1)

    # Plot Cα atoms colored by activation
    sizes = 6 + 50 * acts_3d  # Bigger for higher activation
    ax_3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                 c=rgba, s=sizes, edgecolors="none", zorder=2, depthshade=True)

    # Label top-activating residues on 3D structure
    top_3d = np.where(acts_3d > 0.65)[0]
    if len(top_3d) > 8:
        top_3d = top_3d[np.argsort(-acts_3d[top_3d])][:8]
    for pos in top_3d:
        if pos < len(seq):
            ax_3d.text(coords[pos, 0], coords[pos, 1], coords[pos, 2],
                      seq[pos], fontsize=5.5, fontweight="bold", color=MAGENTA,
                      ha="center", va="bottom", zorder=3)

    # Clean axes
    ax_3d.set_xticklabels([])
    ax_3d.set_yticklabels([])
    ax_3d.set_zticklabels([])
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor("white")
    ax_3d.yaxis.pane.set_edgecolor("white")
    ax_3d.zaxis.pane.set_edgecolor("white")
    ax_3d.grid(False)
    ax_3d.set_axis_off()
    ax_3d.view_init(elev=20, azim=45)


def generate_multi_panel_figure(panels, out_path, suptitle=""):
    """Generate a multi-panel figure (InterPLM Fig 1c/d style)."""
    n = len(panels)
    fig = plt.figure(figsize=(14, 3.2 * n))

    gs = GridSpec(n, 2, width_ratios=[1.8, 1], hspace=0.4, wspace=0.05,
                  left=0.06, right=0.98, top=0.95, bottom=0.04)

    for i, panel in enumerate(panels):
        ax_bar = fig.add_subplot(gs[i, 0])
        ax_3d = fig.add_subplot(gs[i, 1], projection="3d")

        plot_feature_panel(
            ax_bar, ax_3d,
            panel["seq"],
            panel["feat_acts"],
            panel["ca_coords"],
            panel["feature_idx"],
            label=panel.get("label", ""),
            metadata=panel.get("metadata"),
        )

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.99)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved {out_path}.png/pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["esm2", "esm3"], required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--condition", default="S", choices=["S", "S_St"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-indices", type=int, nargs="+", default=None,
                        help="Specific feature indices to visualize")
    parser.add_argument("--max-length", type=int, default=400,
                        help="Max protein length for visualization clarity")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size for SAE encoding")
    args = parser.parse_args()

    if args.layer is None:
        args.layer = 24 if args.model == "esm2" else 33

    out_dir = ROOT / "results" / "figures" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    offsets, accs = build_protein_offsets(sequences)
    structure_dir = ROOT / "data" / "scaled" / "structures"

    # Load SAE
    sae, config = load_sae(args.model, args.layer, args.condition)
    log.info(f"SAE: {config.dict_size} features, k={config.k}")

    # Determine H5 path
    if args.model == "esm2":
        h5_path = ROOT / "data" / "activations" / "esm2_scaled" / f"layer_{args.layer}.h5"
    else:
        prefix = "S" if args.condition == "S" else "S+St"
        h5_path = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / f"{prefix}_layer_{args.layer}.h5"

    # Select features
    if args.feature_indices:
        selected = [(fi, "user_selected", f"Feature {fi}") for fi in args.feature_indices]
    else:
        selected = select_features_from_annotations(args.model, args.layer, args.condition)

    log.info(f"Selected {len(selected)} features:")
    for fi, cat, label in selected:
        log.info(f"  [{cat}] f/{fi}: {label}")

    # Batch-encode all features at once (much faster than per-protein)
    feature_indices = [fi for fi, _, _ in selected]
    log.info(f"\nBatch-encoding {len(feature_indices)} features across all residues...")
    precomputed = batch_encode_feature(sae, h5_path, feature_indices, batch_size=args.batch_size)

    # Generate panels
    panels = []
    for fi, cat, label in selected:
        log.info(f"\nProcessing f/{fi} ({label})...")

        best_acc, best_acts, best_max = find_top_protein_from_precomputed(
            precomputed[fi], offsets, accs, sequences, structure_dir,
            max_length=args.max_length
        )

        if best_acc is None or best_acts is None:
            log.warning(f"  No activating protein found for f/{fi}")
            continue

        log.info(f"  Best protein: {best_acc} (max={best_max:.1f}, len={len(sequences[best_acc])})")

        # Get structure
        pdb_path = structure_dir / f"{best_acc}.pdb"
        ca_coords = parse_pdb_ca(pdb_path) if pdb_path.exists() else np.zeros((0, 3))

        # Get metadata
        prot_meta = metadata.get(best_acc, {})
        prot_name = prot_meta.get("protein_name", best_acc)
        organism = prot_meta.get("organism", "")
        if len(prot_name) > 45:
            prot_name = prot_name[:42] + "..."

        model_label = "ESM-2" if args.model == "esm2" else "ESM-3"
        panel_label = (f"{model_label} L{args.layer} f/{fi}: {label}\n"
                      f"{best_acc} — {prot_name} ({organism})")

        panels.append({
            "seq": sequences[best_acc],
            "feat_acts": best_acts,
            "ca_coords": ca_coords,
            "feature_idx": fi,
            "label": panel_label,
            "metadata": prot_meta,
        })

    if not panels:
        log.error("No panels generated!")
        return

    # Generate figure
    cond_label = f"_{args.condition}" if args.model == "esm3" else ""
    model_label = "ESM-2" if args.model == "esm2" else "ESM-3"

    out_path = out_dir / f"sae_features_{args.model}_L{args.layer}{cond_label}"
    generate_multi_panel_figure(
        panels, str(out_path),
        suptitle=f"SAE Feature Activations — {model_label} Layer {args.layer}"
    )

    # Save feature info as JSON for reference
    info = []
    for panel in panels:
        info.append({
            "feature_idx": panel["feature_idx"],
            "protein": panel["label"].split("\n")[1].split(" — ")[0] if "\n" in panel["label"] else "",
            "max_activation": float(panel["feat_acts"].max()),
            "n_active_residues": int((panel["feat_acts"] > 0).sum()),
            "protein_length": len(panel["seq"]),
        })
    info_path = out_dir / f"sae_features_{args.model}_L{args.layer}{cond_label}_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    log.info(f"\nDone! Generated {len(panels)} feature panels.")


if __name__ == "__main__":
    main()
