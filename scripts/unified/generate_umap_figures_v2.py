#!/usr/bin/env python3
"""Generate UMAP visualizations — v2 with proper feature classification.

1. SAE decoder UMAP for ESM-2 and ESM-3
2. Dataset UMAP for proteins

./env/bin/python scripts/unified/generate_umap_figures_v2.py
"""

import sys
import os
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import numpy as np
import torch
import h5py
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sae.model import TopKSAE

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("umap")


# ── Colors ──
COLORS = {
    "AA-specific":    "#E41A1C",
    "SS-helix (H)":   "#377EB8",
    "SS-strand (E)":  "#4DAF4A",
    "SS-coil (C)":    "#984EA3",
    "Functional":     "#FF7F00",
    "Uncharacterized":"#D0D0D0",
    # Dataset
    "Enzymes":        "#E41A1C",
    "Signaling":      "#377EB8",
    "Structural":     "#4DAF4A",
    "Transport":      "#984EA3",
    "Immune":         "#FF7F00",
    "Transcription":  "#F781BF",
    "Other":          "#BBBBBB",
}


def build_offsets(sequences):
    accs = sorted(sequences.keys())
    offsets = {}
    idx = 0
    for acc in accs:
        offsets[acc] = (idx, len(sequences[acc]))
        idx += len(sequences[acc])
    return offsets, accs


def classify_all_features(sae, h5_path, sequences, offsets, accs,
                          sample_proteins=400):
    """Classify ALL SAE features by type using activation statistics.

    Categories:
      - Dead: <10 activations in sample
      - AA-specific: top AA enrichment > 4x
      - SS-helix/strand/coil: top SS enrichment > 1.3x
      - Functional: functional site fraction > 5%
      - Uncharacterized: everything else
    """
    dict_size = sae.config.dict_size
    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}
    SS_LIST = ["H", "E", "C"]

    rng = np.random.RandomState(42)
    sample = list(rng.choice(accs, min(sample_proteins, len(accs)), replace=False))

    feat_count = np.zeros(dict_size)
    feat_aa_weighted = np.zeros((dict_size, 20))
    feat_ss_weighted = np.zeros((dict_size, 3))

    # Load DSSP annotations
    dssp_path = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"
    dssp_data = {}
    if dssp_path.exists():
        with open(dssp_path) as f:
            dssp_data = json.load(f)  # acc -> list of {resnum, aa, ss3, ss8, asa}
        log.info(f"  Loaded DSSP for {len(dssp_data)} proteins")

    # Load functional site annotations
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    func_sites = {}  # acc -> set of 0-indexed functional positions
    for acc in sample:
        m = metadata.get(acc, {})
        sites = set()
        for feat in m.get("features", []):
            if feat.get("type") in ("Active site", "Binding site", "Disulfide bond"):
                start = feat["start"] - 1
                end = feat.get("end", start + 1)
                for p in range(start, end):
                    sites.add(p)
        if sites:
            func_sites[acc] = sites

    feat_func_weighted = np.zeros(dict_size)
    feat_total_for_func = np.zeros(dict_size)

    log.info(f"Classifying features from {len(sample)} proteins...")
    log.info(f"  DSSP available for {len(dssp_data)} proteins")
    log.info(f"  Functional sites for {len(func_sites)} proteins")

    with h5py.File(h5_path, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]

        for pi, acc in enumerate(sample):
            start, L = offsets[acc]
            seq = sequences[acc]
            L = min(len(seq), L)
            if start + L > total:
                continue

            raw = torch.tensor(acts[start:start+L], dtype=torch.float32)
            with torch.no_grad():
                z = sae.encode(raw).numpy()  # (L, dict_size)

            active = z > 0
            feat_count += active.sum(axis=0)

            # AA distribution
            for pos in range(L):
                if pos < len(seq) and seq[pos] in aa_to_idx:
                    aa_idx = aa_to_idx[seq[pos]]
                    feat_aa_weighted[:, aa_idx] += z[pos] * active[pos]

            # SS distribution (from DSSP)
            # dssp_data[acc] = list of {resnum, aa, ss3, ss8, asa}
            dssp_residues = dssp_data.get(acc, [])
            if dssp_residues and len(dssp_residues) >= L:
                for pos in range(L):
                    ss3 = dssp_residues[pos].get("ss3", "C")
                    if ss3 == "H":
                        ss_idx = 0  # helix
                    elif ss3 == "E":
                        ss_idx = 1  # strand
                    else:
                        ss_idx = 2  # coil
                    feat_ss_weighted[:, ss_idx] += z[pos] * active[pos]

            # Functional site fraction
            sites = func_sites.get(acc, set())
            if sites:
                for pos in range(L):
                    if pos in sites:
                        feat_func_weighted += z[pos] * active[pos]
                    feat_total_for_func += z[pos] * active[pos]

            if (pi + 1) % 100 == 0:
                log.info(f"  {pi+1}/{len(sample)} proteins")

    # Background distributions
    bg_aa = feat_aa_weighted.sum(axis=0)
    bg_aa = bg_aa / (bg_aa.sum() + 1e-10)
    bg_ss = feat_ss_weighted.sum(axis=0)
    bg_ss = bg_ss / (bg_ss.sum() + 1e-10)

    # Classify each feature
    labels = []
    for fi in range(dict_size):
        if feat_count[fi] < 10:
            labels.append("Dead")
            continue

        # AA enrichment
        aa_dist = feat_aa_weighted[fi]
        aa_norm = aa_dist / (aa_dist.sum() + 1e-10)
        aa_enr = aa_norm / (bg_aa + 1e-10)
        max_aa_enr = aa_enr.max()

        # SS enrichment
        ss_dist = feat_ss_weighted[fi]
        ss_total = ss_dist.sum()
        if ss_total > 0:
            ss_norm = ss_dist / ss_total
            ss_enr = ss_norm / (bg_ss + 1e-10)
            max_ss_enr = ss_enr.max()
            ss_pref = int(np.argmax(ss_enr))
        else:
            max_ss_enr = 0
            ss_pref = -1

        # Functional fraction
        func_frac = feat_func_weighted[fi] / (feat_total_for_func[fi] + 1e-10)

        # Priority: AA > Functional > SS > Uncharacterized
        if max_aa_enr > 4.0:
            labels.append("AA-specific")
        elif func_frac > 0.08:
            labels.append("Functional")
        elif max_ss_enr > 1.3 and ss_total > 50:
            if ss_pref == 0:
                labels.append("SS-helix (H)")
            elif ss_pref == 1:
                labels.append("SS-strand (E)")
            else:
                labels.append("SS-coil (C)")
        else:
            labels.append("Uncharacterized")

    return labels


def categorize_protein(meta):
    """Broad functional category for a protein."""
    cat = meta.get("category", "").lower()
    ec = meta.get("ec_numbers", [])
    keywords = set(kw.lower() for kw in meta.get("keywords", []))
    name = meta.get("protein_name", "").lower()

    if "enzyme" in cat or ec:
        return "Enzymes"
    if any(k in cat for k in ["signal", "kinase"]):
        return "Signaling"
    if any(k in cat for k in ["transport", "channel"]):
        return "Transport"
    if any(k in cat for k in ["immune", "immuno"]):
        return "Immune"
    if any(k in cat for k in ["transcript", "dna-binding"]):
        return "Transcription"
    if any(k in cat for k in ["structural", "cytoskel"]):
        return "Structural"

    # Keyword fallback
    enzyme_kw = {"kinase", "phosphatase", "transferase", "hydrolase",
                 "oxidoreductase", "ligase", "synthase", "protease"}
    if keywords & enzyme_kw:
        return "Enzymes"
    if keywords & {"receptor", "signal transduction", "g-protein"}:
        return "Signaling"
    if keywords & {"transport", "ion channel"}:
        return "Transport"
    if keywords & {"immunity", "immunoglobulin", "antigen"}:
        return "Immune"
    if keywords & {"transcription", "dna-binding", "repressor"}:
        return "Transcription"

    # Name-based
    if any(w in name for w in ["kinase", "phosphatase", "synthase", "transferase",
                                "dehydrogenase", "protease", "ligase"]):
        return "Enzymes"
    if "receptor" in name:
        return "Signaling"

    return "Other"


def plot_decoder_umap(W_dec, labels, title, out_path):
    """UMAP of decoder vectors colored by feature type."""
    # Normalize
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_norm = W_dec / (norms + 1e-10)

    # Filter dead
    alive = [i for i, l in enumerate(labels) if l != "Dead"]
    alive_labels = [labels[i] for i in alive]
    W_alive = W_norm[alive]

    log.info(f"  Alive: {len(alive)}/{len(labels)}")
    log.info(f"  Types: {Counter(alive_labels)}")

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.25, metric="cosine",
                        random_state=42, n_jobs=-1)
    emb = reducer.fit_transform(W_alive)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    # Plot order: background first, specific types on top
    order = ["Uncharacterized", "SS-coil (C)", "SS-helix (H)", "SS-strand (E)",
             "Functional", "AA-specific"]

    for cat in order:
        mask = np.array([l == cat for l in alive_labels])
        if mask.sum() == 0:
            continue
        color = COLORS.get(cat, "#999")
        is_bg = cat == "Uncharacterized"
        ax.scatter(emb[mask, 0], emb[mask, 1],
                  c=color, s=2 if is_bg else 14,
                  alpha=0.12 if is_bg else 0.75,
                  zorder=1 if is_bg else 2,
                  label=f"{cat} ({mask.sum()})", edgecolors="none")

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, markerscale=2.5, framealpha=0.85, loc="best",
             handletextpad=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {out_path}")
    return emb, alive, alive_labels


def plot_dataset_umap(h5_path, sequences, metadata, offsets, accs, title, out_path):
    """UMAP of proteins in embedding space colored by function."""
    log.info("  Computing mean-pooled embeddings...")
    embeddings, valid_accs, categories = [], [], []

    with h5py.File(h5_path, "r") as f:
        acts = f["activations"]
        total = acts.shape[0]
        for acc in accs:
            start, L = offsets[acc]
            if start + L > total:
                continue
            embeddings.append(np.mean(acts[start:start+L], axis=0))
            valid_accs.append(acc)
            categories.append(categorize_protein(metadata.get(acc, {})))

    embeddings = np.array(embeddings)
    log.info(f"  Proteins: {len(valid_accs)}, Categories: {Counter(categories)}")

    reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                        random_state=42, n_jobs=-1)
    emb = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    order = ["Other", "Structural", "Transport", "Immune",
             "Transcription", "Signaling", "Enzymes"]
    for cat in order:
        mask = np.array([c == cat for c in categories])
        if mask.sum() == 0:
            continue
        color = COLORS.get(cat, "#999")
        is_bg = cat == "Other"
        ax.scatter(emb[mask, 0], emb[mask, 1],
                  c=color, s=4 if is_bg else 8,
                  alpha=0.25 if is_bg else 0.55,
                  zorder=1 if is_bg else 2,
                  label=f"{cat} ({mask.sum()})", edgecolors="none")

    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7.5, markerscale=2, framealpha=0.85, loc="best",
             handletextpad=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}.{ext}", dpi=250, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {out_path}")
    return emb


def main():
    out_dir = ROOT / "results" / "figures" / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "data" / "scaled" / "sequences" / "sequences.json") as f:
        sequences = json.load(f)
    with open(ROOT / "data" / "scaled" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)
    offsets, accs = build_offsets(sequences)

    # ── 1. ESM-2 Decoder UMAP ──
    log.info("=== ESM-2 Decoder UMAP ===")
    esm2_ckpt_path = ROOT / "models" / "sae" / "esm2_scaled" / "layer_24_topk" / "best.pt"
    esm2_h5 = ROOT / "data" / "activations" / "esm2_scaled" / "layer_24.h5"

    ckpt = torch.load(str(esm2_ckpt_path), map_location="cpu", weights_only=False)
    esm2_sae = TopKSAE(ckpt["sae_config"])
    esm2_sae.load_state_dict(ckpt["model_state_dict"])
    esm2_sae.eval()
    W_dec_esm2 = esm2_sae.decoder.weight.data.T.numpy()

    esm2_labels = classify_all_features(
        esm2_sae, str(esm2_h5), sequences, offsets, accs, sample_proteins=400
    )
    plot_decoder_umap(W_dec_esm2, esm2_labels,
                     "ESM-2 L24 — SAE Feature Space (decoder vectors)",
                     str(out_dir / "umap_decoder_esm2_L24"))

    # ── 2. ESM-3 Decoder UMAP ──
    log.info("\n=== ESM-3 Decoder UMAP ===")
    esm3_ckpt_path = ROOT / "models" / "sae" / "esm3" / "S_layer_33_topk" / "best.pt"
    esm3_h5 = ROOT / "data" / "activations" / "esm3_scaled_multimodal" / "S_layer_33.h5"

    ckpt = torch.load(str(esm3_ckpt_path), map_location="cpu", weights_only=False)
    esm3_sae = TopKSAE(ckpt["sae_config"])
    esm3_sae.load_state_dict(ckpt["model_state_dict"])
    esm3_sae.eval()
    W_dec_esm3 = esm3_sae.decoder.weight.data.T.numpy()

    esm3_labels = classify_all_features(
        esm3_sae, str(esm3_h5), sequences, offsets, accs, sample_proteins=400
    )
    plot_decoder_umap(W_dec_esm3, esm3_labels,
                     "ESM-3 L33 — SAE Feature Space (decoder vectors)",
                     str(out_dir / "umap_decoder_esm3_L33"))

    # ── 3. Dataset UMAP ──
    log.info("\n=== Dataset UMAP ===")
    plot_dataset_umap(str(esm2_h5), sequences, metadata, offsets, accs,
                     "Protein Dataset (4,793 proteins, ESM-2 L24 embeddings)",
                     str(out_dir / "umap_dataset_esm2"))

    log.info("\nDone!")


if __name__ == "__main__":
    main()
