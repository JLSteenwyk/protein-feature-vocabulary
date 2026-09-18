#!/usr/bin/env python3
"""Generate publication-quality InterPLM-style SAE feature figures.

Creates a curated figure with 3 rows per model, each showing:
  - Per-residue activation bar chart with AA letters at peaks
  - 3D protein structure colored cyan→pink by activation

Run as:
    ./env/bin/python scripts/unified/generate_interplm_figure.py

Output: results/figures/unified/fig_sae_features_interplm.{png,pdf}
"""

import sys
import os
import json
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
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBParser

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("interplm")


# ── Colors ──
CYAN = np.array([0.0, 0.737, 0.831])
PINK = np.array([0.906, 0.161, 0.541])
MAGENTA = "#C51B8A"


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


def batch_encode_features(sae, h5_path, feature_indices, batch_size=4096):
    """Batch-encode and extract only the needed feature columns."""
    with h5py.File(h5_path, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]
        result = {fi: np.zeros(total, dtype=np.float32) for fi in feature_indices}
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            raw = torch.tensor(acts[start:end], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw)
            for fi in feature_indices:
                result[fi][start:end] = z[:, fi].numpy()
            if (start // batch_size) % 25 == 0:
                log.info(f"  Encoded {end}/{total}")
    return result


def find_best_protein(feat_acts_full, offsets, accs, sequences, structure_dir,
                      max_length=350):
    best_acc, best_max, best_start, best_L = None, -1, 0, 0
    for acc in accs:
        start, L = offsets[acc]
        if L > max_length or start + L > len(feat_acts_full):
            continue
        if not (structure_dir / f"{acc}.pdb").exists():
            continue
        local_max = float(feat_acts_full[start:start+L].max())
        if local_max > best_max:
            best_max = local_max
            best_acc = acc
            best_start = start
            best_L = L
    if best_acc is None:
        return None, None, -1
    return best_acc, feat_acts_full[best_start:best_start+best_L].copy(), best_max


def parse_ca(pdb_path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("p", str(pdb_path))
    coords = []
    for model in s:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_vector().get_array())
        break
    return np.array(coords)


def draw_panel(ax_bar, ax_3d, seq, feat_acts, ca_coords, title, metadata=None,
               annotate_sites=True):
    """Draw one InterPLM-style panel."""
    L = len(feat_acts)
    max_act = feat_acts.max()
    norm = feat_acts / max_act if max_act > 0 else feat_acts.copy()

    # Bar colors: cyan→pink
    colors = np.array([(1 - a) * CYAN + a * PINK for a in norm])
    ax_bar.bar(np.arange(L), norm, width=1.0, color=colors, edgecolor="none")

    # AA letters at peaks
    threshold = 0.5
    peaks = np.where(norm > threshold)[0]
    if len(peaks) > 18:
        peaks = peaks[np.argsort(-norm[peaks])][:18]
        peaks = sorted(peaks)
    for pos in peaks:
        if pos < len(seq):
            ax_bar.text(pos, norm[pos] + 0.02, seq[pos],
                       ha="center", va="bottom", fontsize=6, fontweight="bold",
                       color=MAGENTA)

    # Annotate functional sites
    if annotate_sites and metadata:
        for feat in metadata.get("features", []):
            ftype = feat.get("type", "")
            if ftype in ("Active site", "Binding site", "Disulfide bond"):
                start = feat["start"] - 1
                end = feat.get("end", start + 1)
                if 0 <= start < L:
                    ax_bar.axvspan(start - 0.5, min(end, L) - 0.5,
                                  alpha=0.08, color="gray", zorder=0)

    ax_bar.set_xlim(-1, L + 1)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_xlabel("Sequence position", fontsize=8)
    ax_bar.set_ylabel("Feature activation", fontsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.tick_params(labelsize=7)
    ax_bar.set_title(title, fontsize=8.5, color=MAGENTA, fontweight="bold", loc="left")

    # 3D structure
    n = min(len(ca_coords), L)
    if n < 3:
        ax_3d.set_axis_off()
        return
    coords = ca_coords[:n]
    acts = norm[:n]
    rgba = np.zeros((n, 4))
    for i in range(n):
        rgba[i, :3] = (1 - acts[i]) * CYAN + acts[i] * PINK
        rgba[i, 3] = 0.5 + 0.5 * acts[i]

    ax_3d.plot(coords[:, 0], coords[:, 1], coords[:, 2],
              color=(*CYAN, 0.25), linewidth=0.4, zorder=1)
    sizes = 6 + 50 * acts
    ax_3d.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                 c=rgba, s=sizes, edgecolors="none", zorder=2, depthshade=True)

    top_3d = np.where(acts > 0.65)[0]
    if len(top_3d) > 6:
        top_3d = top_3d[np.argsort(-acts[top_3d])][:6]
    for pos in top_3d:
        if pos < len(seq):
            ax_3d.text(coords[pos, 0], coords[pos, 1], coords[pos, 2],
                      seq[pos], fontsize=5.5, fontweight="bold", color=MAGENTA,
                      ha="center", va="bottom", zorder=3)

    ax_3d.set_xticklabels([])
    ax_3d.set_yticklabels([])
    ax_3d.set_zticklabels([])
    for axis in [ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis]:
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")
    ax_3d.grid(False)
    ax_3d.set_axis_off()
    ax_3d.view_init(elev=20, azim=45)


def main():
    out_dir = ROOT / "results" / "figures" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = ROOT / "data" / "scaled" / "structures"

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    offsets, accs = build_offsets(sequences)

    # ── Define curated feature sets ──
    # ESM-2 Layer 24
    esm2_features = [
        (5778, "Activates on cysteine residues (C, 10.7x enrichment)"),
        (1415, "Activates on phenylalanine residues (F, 18.7x enrichment)"),
        (830,  "Activates on functional sites (active/binding, 3.8x enrichment)"),
    ]

    # ESM-3 Layer 33 (S-only)
    esm3_features = [
        (8917, "Activates on cysteine residues (C, 35.7x enrichment)"),
        (9000, "Activates on beta strands (E, 4.1x SS enrichment)"),
        (778,  "Activates on alpha helices (H, 2.3x SS enrichment)"),
    ]

    # ── Encode features ──
    log.info("=== ESM-2 Layer 24 ===")
    esm2_sae = load_sae(ROOT / "models" / "sae" / "esm2_scaled" / "layer_24_topk" / "best.pt")
    esm2_h5 = ROOT / "data" / "activations" / "esm2_scaled" / "layer_24.h5"
    esm2_fi = [fi for fi, _ in esm2_features]
    esm2_acts = batch_encode_features(esm2_sae, esm2_h5, esm2_fi)

    log.info("=== ESM-3 Layer 33 ===")
    esm3_sae = load_sae(ROOT / "models" / "sae" / "esm3" / "S_layer_33_topk" / "best.pt")
    esm3_h5 = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S_layer_33.h5"
    esm3_fi = [fi for fi, _ in esm3_features]
    esm3_acts = batch_encode_features(esm3_sae, esm3_h5, esm3_fi)

    # ── Build panels ──
    panels = []

    for fi, desc in esm2_features:
        acc, acts, mx = find_best_protein(esm2_acts[fi], offsets, accs, sequences, structure_dir)
        if acc is None:
            continue
        m = metadata.get(acc, {})
        pname = m.get("protein_name", acc)
        if len(pname) > 40:
            pname = pname[:37] + "..."
        org = m.get("organism", "")
        title = f"ESM-2 L24 f/{fi}: {desc}\n{acc} — {pname} ({org})"
        ca = parse_ca(structure_dir / f"{acc}.pdb")
        panels.append(("esm2", {
            "seq": sequences[acc], "acts": acts, "ca": ca,
            "title": title, "meta": m,
        }))
        log.info(f"  f/{fi}: {acc} (max={mx:.1f})")

    for fi, desc in esm3_features:
        acc, acts, mx = find_best_protein(esm3_acts[fi], offsets, accs, sequences, structure_dir)
        if acc is None:
            continue
        m = metadata.get(acc, {})
        pname = m.get("protein_name", acc)
        if len(pname) > 40:
            pname = pname[:37] + "..."
        org = m.get("organism", "")
        title = f"ESM-3 L33 f/{fi}: {desc}\n{acc} — {pname} ({org})"
        ca = parse_ca(structure_dir / f"{acc}.pdb")
        panels.append(("esm3", {
            "seq": sequences[acc], "acts": acts, "ca": ca,
            "title": title, "meta": m,
        }))
        log.info(f"  f/{fi}: {acc} (max={mx:.1f})")

    # ── Draw figure ──
    n = len(panels)
    fig = plt.figure(figsize=(14, 3.0 * n + 0.8))
    gs = GridSpec(n, 2, width_ratios=[1.8, 1], hspace=0.45, wspace=0.05,
                  left=0.06, right=0.98, top=0.94, bottom=0.03)

    for i, (model, p) in enumerate(panels):
        ax_bar = fig.add_subplot(gs[i, 0])
        ax_3d = fig.add_subplot(gs[i, 1], projection="3d")
        draw_panel(ax_bar, ax_3d, p["seq"], p["acts"], p["ca"],
                  p["title"], p["meta"])

        # Add panel letter
        ax_bar.text(-0.08, 1.15, chr(ord('a') + i),
                   transform=ax_bar.transAxes, fontsize=12,
                   fontweight="bold", va="top")

    fig.suptitle("Representative SAE Feature Activations", fontsize=14,
                fontweight="bold", y=0.98)

    # Add a thin line separating ESM-2 and ESM-3 sections
    n_esm2 = len(esm2_features)
    if n_esm2 < n:
        y_sep = 1.0 - (n_esm2 / n)
        fig.add_artist(plt.Line2D([0.03, 0.97], [y_sep + 0.01, y_sep + 0.01],
                                  transform=fig.transFigure, color="gray",
                                  linewidth=0.5, alpha=0.5))

    for ext in ("png", "pdf"):
        path = out_dir / f"fig_sae_features_interplm.{ext}"
        fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info(f"\nSaved to {out_dir}/fig_sae_features_interplm.png/pdf")


if __name__ == "__main__":
    main()
