#!/usr/bin/env python3
"""Phase 0 Scaled: Full ESM-2 pipeline on 4,793-protein dataset.

Run as: ./env/bin/python scripts/run_phase0_scaled.py

This re-runs Phase 0 (ESM-2 positive control) on the same scaled dataset
used for ESM-3 Phase 1, enabling fair cross-model comparison.

Steps:
  1. Extract ESM-2 activations at 9 layers (GPU)
  2. Linear probing: AA identity, functional sites, SS3, RSA (CPU)
  3. Train TopK SAEs at layers 16 and 24 (GPU)
  4. Feature annotation + DSSP structural analysis (CPU)
  5. ESM-2 ↔ ESM-3 cross-model comparison on matched dataset (CPU)
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from collections import Counter, defaultdict

# Setup paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import torch
import numpy as np
import h5py

# Setup logging
log_dir = ROOT / "results" / "phase0_scaled"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "phase0_scaled_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("phase0_scaled")

# Constants
LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32]
MAX_LENGTH = 600
HIDDEN_DIM = 1280  # ESM-2 650M

# Max ASA (Tien et al. 2013) for RSA computation
MAX_ASA = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 225.0, "E": 223.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

SS8_TO_SS3 = {
    "H": "H", "G": "H", "I": "H",
    "E": "E", "B": "E",
    "T": "C", "S": "C", " ": "C", "C": "C", "P": "C",
}


# ============================================================
# Step 1: Extract activations
# ============================================================
def step1_extract_activations():
    log.info("=" * 60)
    log.info("STEP 1: Extracting ESM-2 650M activations (scaled dataset)")
    log.info("=" * 60)

    output_dir = ROOT / "data" / "activations" / "esm2_scaled"
    index_path = output_dir / "residue_index.json"

    # Idempotency: check if all layer files and index exist
    if index_path.exists():
        all_exist = all((output_dir / f"layer_{l}.h5").exists() for l in LAYERS)
        if all_exist:
            with open(index_path) as f:
                residue_index = json.load(f)
            log.info(f"Step 1 SKIPPED: all outputs exist ({len(residue_index)} proteins)")
            return

    from models.esm2_hooks import load_esm2, ESM2HookManager

    # Load sequences
    seq_path = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
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

    log.info(f"Extracting layers: {LAYERS}")

    output_dir.mkdir(parents=True, exist_ok=True)

    hook_mgr = ESM2HookManager(model, layers=LAYERS, extract_attention=False)
    batch_size = 8

    all_activations = {l: [] for l in LAYERS}
    all_accessions = []
    all_lengths = []

    start_time = time.time()
    skipped = 0
    for i in range(0, len(sequences), batch_size):
        batch_accs = accessions[i : i + batch_size]
        batch_seqs = sequences[i : i + batch_size]

        # Filter long sequences
        valid = [(acc, seq) for acc, seq in zip(batch_accs, batch_seqs) if len(seq) <= MAX_LENGTH]
        if not valid:
            skipped += len(batch_accs)
            continue
        skipped += len(batch_accs) - len(valid)
        valid_accs, valid_seqs = zip(*valid)

        inputs = tokenizer(
            list(valid_seqs), return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH
        ).to(device)

        hook_mgr.cache.clear()
        hook_mgr.register()

        with torch.no_grad():
            model(**inputs)

        for seq_idx in range(len(valid_seqs)):
            seq_len = len(valid_seqs[seq_idx])
            all_accessions.append(valid_accs[seq_idx])
            all_lengths.append(seq_len)
            for layer_idx in LAYERS:
                act = hook_mgr.cache.residual_stream[layer_idx][seq_idx, 1 : seq_len + 1].cpu()
                all_activations[layer_idx].append(act)

        hook_mgr.remove()

        if (i // batch_size) % 50 == 0:
            elapsed = time.time() - start_time
            done = min(i + batch_size, len(sequences))
            rate = done / elapsed if elapsed > 0 else 0
            log.info(f"  Processed {done}/{len(sequences)} sequences ({rate:.1f} seq/s)")

    # Save to HDF5
    for layer_idx in LAYERS:
        layer_acts = torch.cat(all_activations[layer_idx], dim=0)
        h5_path = output_dir / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(), compression="gzip", compression_opts=4)
            f.attrs["layer"] = layer_idx
            f.attrs["num_proteins"] = len(all_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
            f.attrs["hidden_dim"] = layer_acts.shape[1]
        log.info(f"  Saved layer {layer_idx}: {layer_acts.shape} -> {h5_path}")

    # Save residue index
    index = []
    offset = 0
    for acc, length in zip(all_accessions, all_lengths):
        index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - start_time
    log.info(f"Step 1 complete: {len(all_accessions)} proteins, {offset} residues, "
             f"{skipped} skipped (>{MAX_LENGTH} AA) in {elapsed:.0f}s")

    del model
    torch.cuda.empty_cache()


# ============================================================
# Step 2: Linear Probing (AA, functional, SS3, RSA)
# ============================================================
def step2_linear_probing():
    log.info("=" * 60)
    log.info("STEP 2: Linear probing (AA identity, functional sites, SS3, RSA)")
    log.info("=" * 60)

    results_path = ROOT / "results" / "phase0_scaled" / "probing" / "probing_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotency
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        log.info(f"Step 2 SKIPPED: results exist ({len(existing)} probe results)")
        return

    from probing.linear_probe import train_classification_probe, train_regression_probe

    act_dir = ROOT / "data" / "activations" / "esm2_scaled"
    seq_path = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    meta_path = ROOT / "data" / "scaled" / "annotations" / "metadata.json"
    dssp_path = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"

    with open(seq_path) as f:
        seq_dict = json.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)
    with open(dssp_path) as f:
        dssp_annotations = json.load(f)
    with open(act_dir / "residue_index.json") as f:
        residue_index = json.load(f)

    # Build per-residue labels
    log.info("Building per-residue labels...")
    aa_to_class = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    ss3_to_class = {"H": 0, "E": 1, "C": 2}

    aa_labels = []
    functional_labels = []
    ss3_labels = []
    rsa_values = []

    for entry in residue_index:
        acc = entry["accession"]
        seq = seq_dict[acc]
        prot = metadata.get(acc, {})
        dssp = dssp_annotations.get(acc, [])

        for pos in range(entry["length"]):
            # AA identity
            aa = seq[pos] if pos < len(seq) else "X"
            aa_labels.append(aa_to_class.get(aa, -1))

            # Functional site
            is_functional = 0
            for feat in prot.get("features", []):
                start = feat.get("start", 0) - 1  # 1-indexed to 0-indexed
                end = feat.get("end", 0)
                if start <= pos < end:
                    is_functional = 1
                    break
            functional_labels.append(is_functional)

            # SS3 and RSA from DSSP
            if pos < len(dssp):
                res = dssp[pos]
                ss3 = res.get("ss3", "C")
                ss3_labels.append(ss3_to_class.get(ss3, 2))

                asa = res.get("asa", None)
                if asa is not None:
                    max_asa = MAX_ASA.get(aa, 200.0)
                    rsa = min(asa / max_asa, 1.5)
                    rsa_values.append(rsa)
                else:
                    rsa_values.append(np.nan)
            else:
                ss3_labels.append(-1)  # missing
                rsa_values.append(np.nan)

    aa_labels = np.array(aa_labels)
    functional_labels = np.array(functional_labels)
    ss3_labels = np.array(ss3_labels)
    rsa_values = np.array(rsa_values, dtype=np.float32)

    # Verify label count matches HDF5 size (may differ by 1 due to tokenization edge cases)
    with h5py.File(act_dir / f"layer_{LAYERS[0]}.h5", "r") as f:
        n_hdf5 = f["activations"].shape[0]
    if len(aa_labels) != n_hdf5:
        log.warning(f"Label count ({len(aa_labels)}) != HDF5 rows ({n_hdf5}), truncating to match")
        aa_labels = aa_labels[:n_hdf5]
        functional_labels = functional_labels[:n_hdf5]
        ss3_labels = ss3_labels[:n_hdf5]
        rsa_values = rsa_values[:n_hdf5]

    valid_aa_mask = aa_labels >= 0
    valid_ss3_mask = ss3_labels >= 0
    valid_rsa_mask = ~np.isnan(rsa_values) & valid_aa_mask

    log.info(f"Total residues: {len(aa_labels)}")
    log.info(f"Valid AA: {valid_aa_mask.sum()}")
    log.info(f"Functional sites: {functional_labels[valid_aa_mask].sum()} "
             f"({functional_labels[valid_aa_mask].mean()*100:.1f}%)")
    log.info(f"Valid SS3: {valid_ss3_mask.sum()}")
    log.info(f"Valid RSA: {valid_rsa_mask.sum()}")

    # Train/test split (80/20 by proteins)
    n_proteins = len(residue_index)
    n_train = int(0.8 * n_proteins)
    train_residues = sum(e["length"] for e in residue_index[:n_train])
    log.info(f"Train/test split: {n_train}/{n_proteins - n_train} proteins, "
             f"{train_residues}/{len(aa_labels) - train_residues} residues")

    # Subsample for scalability: L-BFGS doesn't need >200K samples to converge
    MAX_TRAIN = 200_000
    MAX_TEST = 50_000
    rng = np.random.default_rng(42)

    results = []

    for layer_idx in LAYERS:
        log.info(f"\n  Layer {layer_idx}:")
        h5_path = act_dir / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "r") as f:
            X_all = f["activations"][:]

        # --- AA identity probe ---
        X = X_all[valid_aa_mask]
        y = aa_labels[valid_aa_mask]
        valid_indices = np.where(valid_aa_mask)[0]
        tr_mask = valid_indices < train_residues
        n_tr = tr_mask.sum()

        X_train_full, X_test_full = X[:n_tr], X[n_tr:]
        y_train_full, y_test_full = y[:n_tr], y[n_tr:]

        # Subsample
        if len(X_train_full) > MAX_TRAIN:
            idx_tr = rng.choice(len(X_train_full), MAX_TRAIN, replace=False)
            X_train, y_train = X_train_full[idx_tr], y_train_full[idx_tr]
        else:
            X_train, y_train = X_train_full, y_train_full
        if len(X_test_full) > MAX_TEST:
            idx_te = rng.choice(len(X_test_full), MAX_TEST, replace=False)
            X_test, y_test = X_test_full[idx_te], y_test_full[idx_te]
        else:
            X_test, y_test = X_test_full, y_test_full

        result = train_classification_probe(
            X_train, y_train, X_test, y_test,
            property_name="amino_acid_identity",
            layer=layer_idx, model_name="esm2_650M_scaled",
        )
        results.append({
            "property": result.property_name, "layer": result.layer,
            "accuracy": result.accuracy, "f1": result.f1, "auroc": result.auroc,
            "task": "classification",
        })
        log.info(f"    AA identity: acc={result.accuracy:.3f} f1={result.f1:.3f}")

        # --- Functional site probe ---
        func_y = functional_labels[valid_aa_mask]
        func_y_train_full, func_y_test_full = func_y[:n_tr], func_y[n_tr:]
        if func_y_test_full.sum() >= 10:
            # Subsample (reuse same indices)
            func_y_train = func_y_train_full[idx_tr] if len(X_train_full) > MAX_TRAIN else func_y_train_full
            func_y_test = func_y_test_full[idx_te] if len(X_test_full) > MAX_TEST else func_y_test_full

            result = train_classification_probe(
                X_train, func_y_train, X_test, func_y_test,
                property_name="functional_site",
                layer=layer_idx, model_name="esm2_650M_scaled",
            )
            results.append({
                "property": result.property_name, "layer": result.layer,
                "accuracy": result.accuracy, "f1": result.f1, "auroc": result.auroc,
                "task": "classification",
            })
            log.info(f"    Functional site: acc={result.accuracy:.3f} auroc={result.auroc:.3f}")

        # --- SS3 probe ---
        ss3_valid = valid_ss3_mask & valid_aa_mask
        X_ss = X_all[ss3_valid]
        y_ss = ss3_labels[ss3_valid]
        ss_valid_indices = np.where(ss3_valid)[0]
        ss_tr_mask = ss_valid_indices < train_residues
        n_ss_tr = ss_tr_mask.sum()

        if n_ss_tr > 100 and (len(y_ss) - n_ss_tr) > 100:
            X_ss_train_full, X_ss_test_full = X_ss[:n_ss_tr], X_ss[n_ss_tr:]
            y_ss_train_full, y_ss_test_full = y_ss[:n_ss_tr], y_ss[n_ss_tr:]

            if len(X_ss_train_full) > MAX_TRAIN:
                idx_ss_tr = rng.choice(len(X_ss_train_full), MAX_TRAIN, replace=False)
                X_ss_train, y_ss_train = X_ss_train_full[idx_ss_tr], y_ss_train_full[idx_ss_tr]
            else:
                X_ss_train, y_ss_train = X_ss_train_full, y_ss_train_full
            if len(X_ss_test_full) > MAX_TEST:
                idx_ss_te = rng.choice(len(X_ss_test_full), MAX_TEST, replace=False)
                X_ss_test, y_ss_test = X_ss_test_full[idx_ss_te], y_ss_test_full[idx_ss_te]
            else:
                X_ss_test, y_ss_test = X_ss_test_full, y_ss_test_full

            result = train_classification_probe(
                X_ss_train, y_ss_train, X_ss_test, y_ss_test,
                property_name="secondary_structure_3",
                layer=layer_idx, model_name="esm2_650M_scaled",
            )
            results.append({
                "property": result.property_name, "layer": result.layer,
                "accuracy": result.accuracy, "f1": result.f1, "auroc": result.auroc,
                "task": "classification",
            })
            log.info(f"    SS3: acc={result.accuracy:.3f} f1={result.f1:.3f}")

        # --- RSA regression probe ---
        rsa_valid = valid_rsa_mask
        X_rsa = X_all[rsa_valid]
        y_rsa = rsa_values[rsa_valid]
        rsa_valid_indices = np.where(rsa_valid)[0]
        rsa_tr_mask = rsa_valid_indices < train_residues
        n_rsa_tr = rsa_tr_mask.sum()

        if n_rsa_tr > 100 and (len(y_rsa) - n_rsa_tr) > 100:
            X_rsa_train_full, X_rsa_test_full = X_rsa[:n_rsa_tr], X_rsa[n_rsa_tr:]
            y_rsa_train_full, y_rsa_test_full = y_rsa[:n_rsa_tr], y_rsa[n_rsa_tr:]

            if len(X_rsa_train_full) > MAX_TRAIN:
                idx_rsa_tr = rng.choice(len(X_rsa_train_full), MAX_TRAIN, replace=False)
                X_rsa_train, y_rsa_train = X_rsa_train_full[idx_rsa_tr], y_rsa_train_full[idx_rsa_tr]
            else:
                X_rsa_train, y_rsa_train = X_rsa_train_full, y_rsa_train_full
            if len(X_rsa_test_full) > MAX_TEST:
                idx_rsa_te = rng.choice(len(X_rsa_test_full), MAX_TEST, replace=False)
                X_rsa_test, y_rsa_test = X_rsa_test_full[idx_rsa_te], y_rsa_test_full[idx_rsa_te]
            else:
                X_rsa_test, y_rsa_test = X_rsa_test_full, y_rsa_test_full

            result = train_regression_probe(
                X_rsa_train, y_rsa_train, X_rsa_test, y_rsa_test,
                property_name="rsa",
                layer=layer_idx, model_name="esm2_650M_scaled",
            )
            results.append({
                "property": result.property_name, "layer": result.layer,
                "mse": result.mse, "task": "regression",
            })
            log.info(f"    RSA: mse={result.mse:.4f}")

        del X_all  # Free memory

    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nStep 2 complete: {len(results)} probe results saved")


# ============================================================
# Step 3: SAE Training (layers 16 and 24)
# ============================================================
def step3_train_saes():
    log.info("=" * 60)
    log.info("STEP 3: Training TopK SAEs on layers 16 and 24")
    log.info("=" * 60)

    from sae.model import SAEConfig
    from sae.train import SAETrainer, TrainConfig

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    target_layers = [16, 24]

    for layer_idx in target_layers:
        output_dir = ROOT / "models" / "sae" / "esm2_scaled" / f"layer_{layer_idx}_topk"
        final_path = output_dir / "final.pt"
        summary_path = ROOT / "results" / "phase0_scaled" / "sae" / f"training_summary_L{layer_idx}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        # Idempotency
        if final_path.exists():
            log.info(f"  Layer {layer_idx} SAE SKIPPED: {final_path} exists")
            continue

        log.info(f"\n  Training SAE for layer {layer_idx}...")

        h5_path = ROOT / "data" / "activations" / "esm2_scaled" / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "r") as f:
            activations = torch.tensor(f["activations"][:], dtype=torch.float32)

        log.info(f"  Loaded activations: {activations.shape}")

        # Train/val split (90/10)
        n = len(activations)
        perm = torch.randperm(n)
        n_train = int(0.9 * n)
        train_acts = activations[perm[:n_train]]
        val_acts = activations[perm[n_train:]]
        log.info(f"  Train: {train_acts.shape}, Val: {val_acts.shape}")

        sae_config = SAEConfig(
            input_dim=HIDDEN_DIM,
            expansion_factor=8,  # 10,240 features
            k=64,
            architecture="topk",
        )
        log.info(f"  SAE config: dict_size={sae_config.dict_size}, K={sae_config.k}")

        train_config = TrainConfig(
            lr=3e-4,
            batch_size=4096,
            num_epochs=15,
            warmup_steps=500,
            log_every=100,
            device=device,
            output_dir=str(output_dir),
        )

        trainer = SAETrainer(sae_config, train_config)
        log.info("  Starting training...")
        summary = trainer.train_on_tensor(train_acts, val_acts)

        trainer.save(final_path)
        log.info(f"  L{layer_idx} done: recon={summary['final_recon']:.6f}, "
                 f"L0={summary['final_l0']:.1f}, dead={summary['dead_frac']:.3f}")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        del activations, train_acts, val_acts, trainer
        torch.cuda.empty_cache()

    log.info("Step 3 complete.")


# ============================================================
# Step 4: Feature Annotation + DSSP Structural Analysis
# ============================================================
def step4_annotate_features():
    log.info("=" * 60)
    log.info("STEP 4: Annotating SAE features with DSSP structural info")
    log.info("=" * 60)

    from sae.model import SAEConfig, build_sae

    # Load shared data
    act_dir = ROOT / "data" / "activations" / "esm2_scaled"
    seq_path = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    meta_path = ROOT / "data" / "scaled" / "annotations" / "metadata.json"
    dssp_path = ROOT / "data" / "scaled" / "annotations" / "dssp_annotations.json"

    with open(seq_path) as f:
        seq_dict = json.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)
    with open(dssp_path) as f:
        dssp_annotations = json.load(f)
    with open(act_dir / "residue_index.json") as f:
        residue_index = json.load(f)

    # Build per-residue labels once
    log.info("Building per-residue annotation arrays...")
    aa_set = list("ACDEFGHIKLMNPQRSTVWY")
    ss3_to_class = {"H": 0, "E": 1, "C": 2}

    aa_labels = []
    functional_labels = []
    ss3_labels = []
    rsa_values = []

    for entry in residue_index:
        acc = entry["accession"]
        seq = seq_dict[acc]
        prot = metadata.get(acc, {})
        dssp = dssp_annotations.get(acc, [])

        for pos in range(entry["length"]):
            aa = seq[pos] if pos < len(seq) else "X"
            aa_labels.append(aa)

            is_func = False
            for feat in prot.get("features", []):
                start = feat.get("start", 0) - 1
                end = feat.get("end", 0)
                if start <= pos < end:
                    is_func = True
                    break
            functional_labels.append(is_func)

            if pos < len(dssp):
                res = dssp[pos]
                ss3_labels.append(res.get("ss3", "C"))
                asa = res.get("asa", None)
                if asa is not None:
                    max_asa = MAX_ASA.get(aa, 200.0)
                    rsa_values.append(min(asa / max_asa, 1.5))
                else:
                    rsa_values.append(np.nan)
            else:
                ss3_labels.append("")
                rsa_values.append(np.nan)

    aa_labels = np.array(aa_labels, dtype=object)
    functional_labels = np.array(functional_labels, dtype=bool)
    ss3_labels = np.array(ss3_labels, dtype=object)
    rsa_values = np.array(rsa_values, dtype=np.float32)

    # Verify label count matches HDF5 size
    with h5py.File(act_dir / f"layer_{LAYERS[0]}.h5", "r") as f:
        n_hdf5 = f["activations"].shape[0]
    if len(aa_labels) != n_hdf5:
        log.warning(f"Label count ({len(aa_labels)}) != HDF5 rows ({n_hdf5}), truncating")
        aa_labels = aa_labels[:n_hdf5]
        functional_labels = functional_labels[:n_hdf5]
        ss3_labels = ss3_labels[:n_hdf5]
        rsa_values = rsa_values[:n_hdf5]

    total_residues = len(aa_labels)
    func_rate = functional_labels.mean()
    ss3_counts = Counter(ss3_labels[ss3_labels != ""])
    log.info(f"Total residues: {total_residues}")
    log.info(f"Functional site rate: {func_rate*100:.1f}%")
    log.info(f"SS3 distribution: H={ss3_counts.get('H',0)}, E={ss3_counts.get('E',0)}, C={ss3_counts.get('C',0)}")

    # Background frequencies
    bg_aa_counts = Counter(aa_labels)
    bg_func_rate = functional_labels.mean()
    bg_ss3_counts = Counter(s for s in ss3_labels if s in ("H", "E", "C"))
    bg_ss3_total = sum(bg_ss3_counts.values())
    bg_ss3_fracs = {ss: bg_ss3_counts.get(ss, 0) / bg_ss3_total for ss in ("H", "E", "C")} if bg_ss3_total > 0 else {}

    target_layers = [16, 24]
    device = "cpu"  # Annotation on CPU

    for layer_idx in target_layers:
        out_path = ROOT / "results" / "phase0_scaled" / "sae" / f"feature_annotations_L{layer_idx}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Idempotency
        if out_path.exists():
            log.info(f"  Layer {layer_idx} annotations SKIPPED: {out_path} exists")
            continue

        model_path = ROOT / "models" / "sae" / "esm2_scaled" / f"layer_{layer_idx}_topk" / "final.pt"
        if not model_path.exists():
            log.warning(f"  Layer {layer_idx} SAE not found at {model_path}, skipping")
            continue

        log.info(f"\n  Annotating layer {layer_idx} SAE features...")

        # Load SAE
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        config = ckpt["sae_config"]
        sae = build_sae(config)
        sae.load_state_dict(ckpt["model_state_dict"])
        sae.eval()

        # Load activations
        h5_path = act_dir / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "r") as f:
            activations = torch.tensor(f["activations"][:], dtype=torch.float32)

        log.info(f"  Encoding {activations.shape[0]} residues through SAE...")

        # Encode all in batches
        all_z = []
        batch_size = 8192
        with torch.no_grad():
            for i in range(0, len(activations), batch_size):
                batch = activations[i : i + batch_size]
                z = sae.encode(batch)
                all_z.append(z.cpu())
        all_z = torch.cat(all_z, dim=0)  # (total_residues, dict_size)

        dict_size = all_z.shape[1]
        feature_freq = (all_z > 0).float().mean(dim=0)
        active_count = (feature_freq > 0.001).sum().item()
        dead_count = (feature_freq == 0).sum().item()
        log.info(f"  Active features (>0.1% freq): {active_count}, Dead: {dead_count}")

        # Annotate top-200 most active features
        top_indices = feature_freq.argsort(descending=True)[:200]

        feature_annotations = []
        for feat_idx in top_indices:
            feat_idx = feat_idx.item()
            feat_acts = all_z[:, feat_idx].numpy()
            active_mask = feat_acts > 0

            if active_mask.sum() < 10:
                continue

            total_active = active_mask.sum()

            # AA enrichment
            active_aas = aa_labels[active_mask]
            aa_fracs = {}
            enrichments = {}
            for aa in aa_set:
                count_active = (active_aas == aa).sum()
                frac = count_active / total_active
                aa_fracs[aa] = frac
                bg_frac = bg_aa_counts.get(aa, 0) / total_residues
                enrichments[aa] = frac / bg_frac if bg_frac > 0 else 0.0
            top_enriched = sorted(enrichments.items(), key=lambda x: -x[1])[:5]

            # Functional site enrichment
            func_in_active = functional_labels[active_mask].mean()
            func_enrichment = func_in_active / bg_func_rate if bg_func_rate > 0 else 0.0

            # SS preference (enrichment of each SS type)
            ss_enrichment = {}
            ss_in_active = ss3_labels[active_mask]
            for ss in ("H", "E", "C"):
                ss_count = (ss_in_active == ss).sum()
                ss_frac = ss_count / total_active
                bg = bg_ss3_fracs.get(ss, 0.33)
                ss_enrichment[ss] = round(ss_frac / bg, 4) if bg > 0 else 0.0
            ss_preference = max(ss_enrichment, key=ss_enrichment.get) if ss_enrichment else "C"
            max_ss_enrichment = ss_enrichment.get(ss_preference, 0.0)

            # RSA correlation
            valid_rsa = ~np.isnan(rsa_values[active_mask])
            if valid_rsa.sum() > 10:
                # Correlation of feature activation with RSA across all valid residues
                all_valid_rsa = ~np.isnan(rsa_values)
                if all_valid_rsa.sum() > 100:
                    r = np.corrcoef(feat_acts[all_valid_rsa], rsa_values[all_valid_rsa])[0, 1]
                    if np.isnan(r):
                        r = 0.0
                else:
                    r = 0.0
            else:
                r = 0.0

            surface_pref = "surface" if r > 0.1 else ("buried" if r < -0.1 else "neutral")

            feature_annotations.append({
                "feature_idx": feat_idx,
                "activation_freq": round(feature_freq[feat_idx].item(), 6),
                "mean_activation": round(float(feat_acts[active_mask].mean()), 4),
                "top_enriched_aa": [(aa, round(e, 2)) for aa, e in top_enriched],
                "functional_site_enrichment": round(func_enrichment, 2),
                "functional_site_frac": round(float(func_in_active), 4),
                "ss_enrichment": ss_enrichment,
                "ss_preference": ss_preference,
                "max_ss_enrichment": round(max_ss_enrichment, 4),
                "rsa_correlation": round(r, 4),
                "surface_preference": surface_pref,
            })

        # Summary
        n_aa_specific = sum(1 for a in feature_annotations
                            if any(e > 3.0 for _, e in a["top_enriched_aa"]))
        n_functional = sum(1 for a in feature_annotations
                           if a["functional_site_enrichment"] > 2.0)
        n_strong_ss = sum(1 for a in feature_annotations
                          if a["max_ss_enrichment"] > 2.0)
        ss_prefs = Counter(a["ss_preference"] for a in feature_annotations)
        surf_prefs = Counter(a["surface_preference"] for a in feature_annotations)

        log.info(f"  Feature summary (L{layer_idx}):")
        log.info(f"    Total annotated: {len(feature_annotations)}")
        log.info(f"    AA-specific (>3x): {n_aa_specific}")
        log.info(f"    Functional (>2x): {n_functional}")
        log.info(f"    Strong SS (>2x): {n_strong_ss}")
        log.info(f"    SS pref: H={ss_prefs.get('H',0)}, E={ss_prefs.get('E',0)}, C={ss_prefs.get('C',0)}")
        log.info(f"    Surface: surf={surf_prefs.get('surface',0)}, buried={surf_prefs.get('buried',0)}, neutral={surf_prefs.get('neutral',0)}")

        with open(out_path, "w") as f:
            json.dump(feature_annotations, f, indent=2)
        log.info(f"  Saved to {out_path}")

        del activations, all_z, sae

    log.info("Step 4 complete.")


# ============================================================
# Step 5: ESM-2 ↔ ESM-3 Cross-Model Comparison
# ============================================================
def step5_cross_model_comparison():
    log.info("=" * 60)
    log.info("STEP 5: ESM-2 ↔ ESM-3 cross-model SAE feature comparison")
    log.info("=" * 60)

    from scipy.optimize import linear_sum_assignment

    out_path = ROOT / "results" / "phase0_scaled" / "cross_model_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Idempotency
    if out_path.exists():
        log.info(f"Step 5 SKIPPED: {out_path} exists")
        return

    AAS = list("ACDEFGHIKLMNPQRSTVWY")

    def build_aa_vector(feature):
        vec = np.ones(20)
        for aa, enrichment in feature.get("top_enriched_aa", []):
            if aa in AAS:
                idx = AAS.index(aa)
                vec[idx] = enrichment
        return vec

    def compute_pairwise_similarity(mat1, mat2):
        norms1 = np.linalg.norm(mat1, axis=1, keepdims=True)
        norms2 = np.linalg.norm(mat2, axis=1, keepdims=True)
        norms1[norms1 == 0] = 1
        norms2[norms2 == 0] = 1
        return (mat1 / norms1) @ (mat2 / norms2).T

    # Load ESM-2 L24 annotations (scaled)
    esm2_path = ROOT / "results" / "phase0_scaled" / "sae" / "feature_annotations_L24.json"
    if not esm2_path.exists():
        log.warning(f"ESM-2 L24 annotations not found at {esm2_path}, skipping step 5")
        return

    with open(esm2_path) as f:
        esm2_ann = json.load(f)
    log.info(f"ESM-2 L24 features: {len(esm2_ann)}")

    # Load ESM-3 S-only L33 annotations (scaled)
    # Try scaled SAE annotations first, fall back to pilot annotations
    esm3_path = ROOT / "results" / "phase1_scaled" / "sae_annotations" / "S_layer_33_annotations.json"
    if not esm3_path.exists():
        # Fall back: try to generate from scaled SAE + activations
        esm3_path = ROOT / "results" / "phase1" / "modality_saes" / "S_layer_33_annotations.json"
    if not esm3_path.exists():
        log.warning(f"ESM-3 S-only L33 annotations not found, skipping step 5")
        return

    with open(esm3_path) as f:
        esm3_ann = json.load(f)
    if isinstance(esm3_ann, dict) and "features" in esm3_ann:
        esm3_ann = esm3_ann["features"]
    log.info(f"ESM-3 S-only L33 features: {len(esm3_ann)} (from {esm3_path.name})")

    # Build AA enrichment matrices
    esm2_mat = np.array([build_aa_vector(f) for f in esm2_ann])
    esm3_mat = np.array([build_aa_vector(f) for f in esm3_ann])

    # Compute pairwise cosine similarity
    sim_matrix = compute_pairwise_similarity(esm2_mat, esm3_mat)
    log.info(f"Similarity matrix: {sim_matrix.shape}, mean={sim_matrix.mean():.4f}")

    # Hungarian matching
    cost = -sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(row_ind, col_ind):
        matches.append({
            "esm2_idx": int(i), "esm3_idx": int(j),
            "similarity": float(sim_matrix[i, j])
        })
    matches.sort(key=lambda x: -x["similarity"])

    match_sims = [m["similarity"] for m in matches]
    log.info(f"Mean match similarity: {np.mean(match_sims):.4f}")
    log.info(f"Median match similarity: {np.median(match_sims):.4f}")

    # Count convergent at various thresholds
    thresholds = [0.90, 0.92, 0.95, 0.97, 0.99]
    threshold_counts = {}
    for t in thresholds:
        n = sum(1 for m in matches if m["similarity"] >= t)
        threshold_counts[str(t)] = n
        pct = 100 * n / len(matches) if matches else 0
        log.info(f"  sim >= {t}: {n}/{len(matches)} ({pct:.1f}%)")

    # Permutation test
    log.info("Running permutation test (1000 shuffles)...")
    rng = np.random.default_rng(42)
    null_mean_sims = []
    for _ in range(1000):
        shuffled = esm3_mat.copy()
        rng.shuffle(shuffled, axis=1)
        null_sim = compute_pairwise_similarity(esm2_mat, shuffled)
        null_cost = -null_sim
        null_row, null_col = linear_sum_assignment(null_cost)
        null_mean = np.mean([null_sim[i, j] for i, j in zip(null_row, null_col)])
        null_mean_sims.append(null_mean)

    observed_mean = np.mean(match_sims)
    null_mean = np.mean(null_mean_sims)
    null_std = np.std(null_mean_sims)
    z_score = (observed_mean - null_mean) / null_std if null_std > 0 else float('inf')
    p_val = np.mean([n >= observed_mean for n in null_mean_sims])

    log.info(f"Observed mean: {observed_mean:.4f}")
    log.info(f"Null mean: {null_mean:.4f} +/- {null_std:.4f}")
    log.info(f"Z-score: {z_score:.2f}, p-value: {p_val:.4f}")

    # Feature type classification
    def classify(feat):
        enrichments = [e for _, e in feat.get("top_enriched_aa", [])]
        max_enrich = max(enrichments) if enrichments else 1.0
        func_enrich = feat.get("functional_site_enrichment", 1.0)
        if max_enrich > 3.0:
            return "aa_specific"
        elif func_enrich > 2.0:
            return "functional"
        return "generic"

    esm2_types = Counter(classify(f) for f in esm2_ann)
    esm3_types = Counter(classify(f) for f in esm3_ann)

    # Convergent features
    convergent = []
    for m in matches:
        if m["similarity"] >= 0.95:
            e2 = esm2_ann[m["esm2_idx"]]
            e3 = esm3_ann[m["esm3_idx"]]
            e2_aas = {aa for aa, enr in e2.get("top_enriched_aa", []) if enr > 2.0}
            e3_aas = {aa for aa, enr in e3.get("top_enriched_aa", []) if enr > 2.0}
            convergent.append({
                "esm2_feature": e2["feature_idx"],
                "esm3_feature": e3["feature_idx"],
                "similarity": m["similarity"],
                "shared_enriched_aa": sorted(e2_aas & e3_aas),
            })

    n_conv = threshold_counts.get("0.95", 0)
    conv_pct = 100 * n_conv / len(matches) if matches else 0

    # Compile results
    results = {
        "comparison": "ESM-2 L24 vs ESM-3 S-only L33 (scaled dataset)",
        "dataset": "4793 proteins (same dataset for both models)",
        "n_esm2_features": len(esm2_ann),
        "n_esm3_features": len(esm3_ann),
        "esm3_annotation_source": str(esm3_path),
        "similarity_stats": {
            "mean": float(np.mean(match_sims)),
            "median": float(np.median(match_sims)),
            "std": float(np.std(match_sims)),
            "min": float(np.min(match_sims)),
            "max": float(np.max(match_sims)),
        },
        "convergent_counts": threshold_counts,
        "convergent_fraction_095": round(conv_pct, 1),
        "permutation_test": {
            "observed_mean": float(observed_mean),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "z_score": float(z_score),
            "p_value": float(p_val),
            "n_permutations": 1000,
        },
        "feature_types": {
            "esm2": dict(esm2_types),
            "esm3": dict(esm3_types),
        },
        "convergent_features": convergent[:20],
        "top_matches": matches[:20],
        "pilot_comparison_note": "Pilot found 87.5% convergence (938 proteins). "
                                  "This scaled run uses 4793 proteins for fair comparison.",
    }

    log.info(f"\n{'=' * 60}")
    log.info("CROSS-MODEL COMPARISON SUMMARY")
    log.info(f"{'=' * 60}")
    log.info(f"Convergent features (>0.95): {n_conv}/{len(matches)} ({conv_pct:.1f}%)")
    log.info(f"Permutation test: z={z_score:.1f}, p={p_val:.4f}")
    log.info(f"ESM-2 feature types: {dict(esm2_types)}")
    log.info(f"ESM-3 feature types: {dict(esm3_types)}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {out_path}")

    log.info("Step 5 complete.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PHASE 0 SCALED: ESM-2 Pipeline on 4,793 Proteins")
    log.info("=" * 60)

    overall_start = time.time()

    try:
        step1_extract_activations()
        step2_linear_probing()
        step3_train_saes()
        step4_annotate_features()
        step5_cross_model_comparison()

        elapsed = time.time() - overall_start
        log.info(f"\nPHASE 0 SCALED COMPLETE in {elapsed/60:.1f} minutes")
        log.info("Results saved to results/phase0_scaled/")

    except Exception as e:
        log.error(f"Phase 0 scaled failed: {e}", exc_info=True)
        raise
