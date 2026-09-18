"""Phase 0: Full pipeline — extraction, probing, SAE training.

Run as: ./env/bin/python scripts/run_phase0.py
Logs to: results/phase0/phase0_log.txt

This script runs the entire Phase 0 pipeline:
1. Extract ESM-2 650M activations on pilot set (5 layers)
2. Train linear probes for SS, SA, binding sites
3. Train TopK SAE on layer 24
4. Annotate SAE features
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

# Setup logging
log_dir = ROOT / "results" / "phase0"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "phase0_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("phase0")

import torch
import numpy as np
import h5py


# ============================================================
# Step 1: Extract activations
# ============================================================
def step1_extract_activations():
    log.info("=" * 60)
    log.info("STEP 1: Extracting ESM-2 650M activations")
    log.info("=" * 60)

    from models.esm2_hooks import load_esm2, ESM2HookManager

    # Load sequences
    seq_path = ROOT / "data" / "pilot" / "sequences" / "sequences.json"
    with open(seq_path) as f:
        seq_dict = json.load(f)

    accessions = list(seq_dict.keys())
    sequences = list(seq_dict.values())
    log.info(f"Loaded {len(sequences)} sequences")

    # Load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading ESM-2 650M on {device}...")
    model, tokenizer = load_esm2("esm2_t33_650M_UR50D", device=device)
    log.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    # Target layers
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 32]
    log.info(f"Extracting layers: {layers}")

    # Extract in batches
    output_dir = ROOT / "data" / "activations" / "esm2"
    output_dir.mkdir(parents=True, exist_ok=True)

    hook_mgr = ESM2HookManager(model, layers=layers, extract_attention=False)
    batch_size = 8
    max_length = 800

    all_activations = {l: [] for l in layers}
    all_accessions = []
    all_lengths = []

    start_time = time.time()
    for i in range(0, len(sequences), batch_size):
        batch_accs = accessions[i : i + batch_size]
        batch_seqs = sequences[i : i + batch_size]

        # Filter long sequences
        valid = [(acc, seq) for acc, seq in zip(batch_accs, batch_seqs) if len(seq) <= max_length]
        if not valid:
            continue
        valid_accs, valid_seqs = zip(*valid)

        inputs = tokenizer(
            list(valid_seqs), return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)

        hook_mgr.cache.clear()
        hook_mgr.register()

        with torch.no_grad():
            model(**inputs)

        # Save per-sequence (excluding BOS/EOS tokens: positions 1:-1)
        for seq_idx in range(len(valid_seqs)):
            seq_len = len(valid_seqs[seq_idx])
            all_accessions.append(valid_accs[seq_idx])
            all_lengths.append(seq_len)
            for layer_idx in layers:
                # Extract positions 1:seq_len+1 (skip BOS, up to but not including EOS)
                act = hook_mgr.cache.residual_stream[layer_idx][seq_idx, 1 : seq_len + 1].cpu()
                all_activations[layer_idx].append(act)

        hook_mgr.remove()

        if (i // batch_size) % 20 == 0:
            elapsed = time.time() - start_time
            done = min(i + batch_size, len(sequences))
            rate = done / elapsed if elapsed > 0 else 0
            log.info(f"  Processed {done}/{len(sequences)} sequences ({rate:.1f} seq/s)")

    # Save to HDF5 — one file per layer with concatenated residue activations
    for layer_idx in layers:
        layer_acts = torch.cat(all_activations[layer_idx], dim=0)  # (total_residues, 1280)
        h5_path = output_dir / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(), compression="gzip", compression_opts=4)
            f.attrs["layer"] = layer_idx
            f.attrs["num_proteins"] = len(all_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
            f.attrs["hidden_dim"] = layer_acts.shape[1]
        log.info(f"  Saved layer {layer_idx}: {layer_acts.shape} -> {h5_path}")

    # Save index mapping residues back to proteins
    index = []
    offset = 0
    for acc, length in zip(all_accessions, all_lengths):
        index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length
    with open(output_dir / "residue_index.json", "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - start_time
    log.info(f"Step 1 complete: {len(all_accessions)} proteins, {offset} residues in {elapsed:.0f}s")

    del model
    torch.cuda.empty_cache()
    return all_accessions, all_lengths


# ============================================================
# Step 2: Linear Probing
# ============================================================
def step2_linear_probing():
    log.info("=" * 60)
    log.info("STEP 2: Linear probing")
    log.info("=" * 60)

    from probing.linear_probe import train_classification_probe, ProbeResult

    act_dir = ROOT / "data" / "activations" / "esm2"
    meta_path = ROOT / "data" / "pilot" / "annotations" / "metadata.json"

    with open(meta_path) as f:
        metadata = json.load(f)

    # Load residue index
    with open(act_dir / "residue_index.json") as f:
        residue_index = json.load(f)

    # Build per-residue labels from metadata
    # For now: simple amino acid identity probing (always available) + functional site labels
    log.info("Building per-residue labels...")

    # Amino acid composition probing (sanity check - should be ~100% from early layers)
    aa_to_class = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    seq_dict_path = ROOT / "data" / "pilot" / "sequences" / "sequences.json"
    with open(seq_dict_path) as f:
        seq_dict = json.load(f)

    aa_labels = []
    functional_site_labels = []  # 1 = active/binding site, 0 = other
    for entry in residue_index:
        acc = entry["accession"]
        seq = seq_dict[acc]
        for pos, aa in enumerate(seq[: entry["length"]]):
            aa_labels.append(aa_to_class.get(aa, -1))

            # Check if this position is a functional site
            is_functional = 0
            prot = metadata.get(acc, {})
            for feat in prot.get("features", []):
                start = feat.get("start", 0) - 1  # 1-indexed to 0-indexed
                end = feat.get("end", 0)  # end is inclusive in UniProt
                if start <= pos < end:
                    is_functional = 1
                    break
            functional_site_labels.append(is_functional)

    aa_labels = np.array(aa_labels)
    functional_site_labels = np.array(functional_site_labels)

    # Filter out unknown AAs
    valid_mask = aa_labels >= 0
    aa_labels = aa_labels[valid_mask]
    functional_mask = functional_site_labels[valid_mask]

    log.info(f"Total residues: {len(aa_labels)}")
    log.info(f"Functional sites: {functional_mask.sum()} ({functional_mask.mean()*100:.1f}%)")

    # Train/test split (80/20 by proteins)
    n_proteins = len(residue_index)
    n_train = int(0.8 * n_proteins)
    train_residues = sum(e["length"] for e in residue_index[:n_train])

    results = []
    layers = [0, 4, 8, 12, 16, 20, 24, 28, 32]

    for layer_idx in layers:
        h5_path = act_dir / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "r") as f:
            X = f["activations"][:][valid_mask]

        X_train, X_test = X[:train_residues], X[train_residues:]
        y_aa_train, y_aa_test = aa_labels[:train_residues], aa_labels[train_residues:]
        y_func_train, y_func_test = functional_mask[:train_residues], functional_mask[train_residues:]

        # AA identity probe (sanity check)
        result = train_classification_probe(
            X_train, y_aa_train, X_test, y_aa_test,
            property_name="amino_acid_identity",
            layer=layer_idx, model_name="esm2_650M",
        )
        results.append(result)
        log.info(f"  Layer {layer_idx} AA identity: acc={result.accuracy:.3f} f1={result.f1:.3f}")

        # Functional site probe (if enough positive examples)
        if y_func_test.sum() >= 10:
            result = train_classification_probe(
                X_train, y_func_train, X_test, y_func_test,
                property_name="functional_site",
                layer=layer_idx, model_name="esm2_650M",
            )
            results.append(result)
            log.info(f"  Layer {layer_idx} functional site: acc={result.accuracy:.3f} f1={result.f1:.3f}")

    # Save results
    results_data = [
        {"property": r.property_name, "layer": r.layer, "accuracy": r.accuracy, "f1": r.f1, "auroc": r.auroc}
        for r in results
    ]
    with open(ROOT / "results" / "phase0" / "probing" / "probing_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    log.info("Step 2 complete.")
    return results


# ============================================================
# Step 3: SAE Training
# ============================================================
def step3_train_sae():
    log.info("=" * 60)
    log.info("STEP 3: Training TopK SAE on layer 24")
    log.info("=" * 60)

    from sae.model import SAEConfig
    from sae.train import SAETrainer, TrainConfig

    # Load layer 24 activations
    h5_path = ROOT / "data" / "activations" / "esm2" / "layer_24.h5"
    with h5py.File(h5_path, "r") as f:
        activations = torch.tensor(f["activations"][:], dtype=torch.float32)

    log.info(f"Loaded activations: {activations.shape}")

    # Split train/val (90/10)
    n = len(activations)
    perm = torch.randperm(n)
    n_train = int(0.9 * n)
    train_acts = activations[perm[:n_train]]
    val_acts = activations[perm[n_train:]]
    log.info(f"Train: {train_acts.shape}, Val: {val_acts.shape}")

    # SAE config
    hidden_dim = activations.shape[1]  # 1280
    sae_config = SAEConfig(
        input_dim=hidden_dim,
        expansion_factor=8,  # 10240 features
        k=64,
        architecture="topk",
    )
    log.info(f"SAE config: dict_size={sae_config.dict_size}, K={sae_config.k}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_config = TrainConfig(
        lr=3e-4,
        batch_size=4096,
        num_epochs=15,
        warmup_steps=500,
        log_every=100,
        device=device,
        output_dir=str(ROOT / "models" / "sae" / "esm2" / "layer_24_topk"),
    )

    trainer = SAETrainer(sae_config, train_config)
    log.info("Starting SAE training...")
    summary = trainer.train_on_tensor(train_acts, val_acts)

    trainer.save(Path(train_config.output_dir) / "final.pt")
    log.info(f"SAE training complete: recon={summary['final_recon']:.6f}, L0={summary['final_l0']:.1f}, dead={summary['dead_frac']:.3f}")

    # Save training summary
    with open(ROOT / "results" / "phase0" / "sae_reproduction" / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return trainer


# ============================================================
# Step 4: Feature Annotation
# ============================================================
def step4_annotate_features(trainer=None):
    log.info("=" * 60)
    log.info("STEP 4: Annotating SAE features")
    log.info("=" * 60)

    from sae.train import SAETrainer

    if trainer is None:
        model_path = ROOT / "models" / "sae" / "esm2" / "layer_24_topk" / "final.pt"
        trainer = SAETrainer.load(str(model_path), device="cuda:0" if torch.cuda.is_available() else "cpu")

    # Load activations and metadata
    h5_path = ROOT / "data" / "activations" / "esm2" / "layer_24.h5"
    with h5py.File(h5_path, "r") as f:
        activations = torch.tensor(f["activations"][:], dtype=torch.float32)

    with open(ROOT / "data" / "activations" / "esm2" / "residue_index.json") as f:
        residue_index = json.load(f)

    with open(ROOT / "data" / "pilot" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    with open(ROOT / "data" / "pilot" / "sequences" / "sequences.json") as f:
        seq_dict = json.load(f)

    # Encode all activations through SAE
    log.info("Encoding activations through SAE...")
    device = trainer.train_config.device
    trainer.sae.eval()

    all_features = []
    batch_size = 8192
    for i in range(0, len(activations), batch_size):
        batch = activations[i : i + batch_size].to(device)
        with torch.no_grad():
            z = trainer.sae.encode(batch)
        all_features.append(z.cpu())
    all_features = torch.cat(all_features, dim=0)  # (total_residues, dict_size)

    log.info(f"Feature matrix: {all_features.shape}")

    # Build per-residue AA labels and functional annotations
    aa_labels = []
    functional_labels = []
    for entry in residue_index:
        acc = entry["accession"]
        seq = seq_dict[acc]
        prot = metadata.get(acc, {})
        for pos in range(entry["length"]):
            aa_labels.append(seq[pos] if pos < len(seq) else "X")

            is_func = False
            for feat in prot.get("features", []):
                start = feat.get("start", 0) - 1
                end = feat.get("end", 0)
                if start <= pos < end:
                    is_func = True
                    break
            functional_labels.append(is_func)

    aa_labels = np.array(aa_labels)
    functional_labels = np.array(functional_labels)

    # Analyze top features
    dict_size = all_features.shape[1]
    feature_activation_freq = (all_features > 0).float().mean(dim=0)
    active_features = (feature_activation_freq > 0.001).sum().item()
    dead_features = (feature_activation_freq == 0).sum().item()

    log.info(f"Active features (>0.1% freq): {active_features}")
    log.info(f"Dead features: {dead_features}")

    # Annotate top-200 most active features
    top_feature_indices = feature_activation_freq.argsort(descending=True)[:200]

    feature_annotations = []
    aa_set = list("ACDEFGHIKLMNPQRSTVWY")

    for feat_idx in top_feature_indices:
        feat_idx = feat_idx.item()
        feat_acts = all_features[:, feat_idx]
        active_mask = feat_acts > 0

        if active_mask.sum() < 10:
            continue

        # AA composition at activating positions
        active_aas = aa_labels[active_mask.numpy()]
        aa_counts = {aa: (active_aas == aa).sum() for aa in aa_set}
        total_active = len(active_aas)
        aa_fracs = {aa: count / total_active for aa, count in aa_counts.items()}

        # Background AA composition
        bg_counts = {aa: (aa_labels == aa).sum() for aa in aa_set}
        total_bg = len(aa_labels)
        bg_fracs = {aa: count / total_bg for aa, count in bg_counts.items()}

        # Find most enriched AAs
        enrichments = {}
        for aa in aa_set:
            if bg_fracs[aa] > 0:
                enrichments[aa] = aa_fracs[aa] / bg_fracs[aa]

        top_enriched = sorted(enrichments.items(), key=lambda x: -x[1])[:3]

        # Functional site enrichment
        func_in_active = functional_labels[active_mask.numpy()].mean()
        func_background = functional_labels.mean()
        func_enrichment = func_in_active / func_background if func_background > 0 else 0

        annotation = {
            "feature_idx": feat_idx,
            "activation_freq": feature_activation_freq[feat_idx].item(),
            "mean_activation": feat_acts[active_mask].mean().item(),
            "top_enriched_aa": [(aa, round(e, 2)) for aa, e in top_enriched],
            "functional_site_enrichment": round(func_enrichment, 2),
            "functional_site_frac": round(func_in_active, 4),
        }
        feature_annotations.append(annotation)

    # Classify features
    ss_features = sum(1 for a in feature_annotations
                      if any(e > 2.5 for _, e in a["top_enriched_aa"]
                             if _ in "GP"))  # Rough proxy: G/P enriched = structural
    func_features = sum(1 for a in feature_annotations
                        if a["functional_site_enrichment"] > 2.0)
    physicochemical = sum(1 for a in feature_annotations
                          if any(e > 2.0 for _, e in a["top_enriched_aa"]))

    log.info(f"\nFeature annotation summary:")
    log.info(f"  Total annotated: {len(feature_annotations)}")
    log.info(f"  With strong AA enrichment: {physicochemical}")
    log.info(f"  With functional site enrichment (>2x): {func_features}")
    log.info(f"  With structural AA enrichment (G/P): {ss_features}")

    # Save annotations
    with open(ROOT / "results" / "phase0" / "sae_reproduction" / "feature_annotations.json", "w") as f:
        json.dump(feature_annotations, f, indent=2)

    log.info("Step 4 complete.")
    return feature_annotations


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PHASE 0: ESM-2 Pilot Pipeline")
    log.info("=" * 60)

    overall_start = time.time()

    try:
        step1_extract_activations()
        step2_linear_probing()
        trainer = step3_train_sae()
        step4_annotate_features(trainer)

        elapsed = time.time() - overall_start
        log.info(f"\nPHASE 0 COMPLETE in {elapsed/60:.1f} minutes")
        log.info("Results saved to results/phase0/")

    except Exception as e:
        log.error(f"Phase 0 failed: {e}", exc_info=True)
        raise
