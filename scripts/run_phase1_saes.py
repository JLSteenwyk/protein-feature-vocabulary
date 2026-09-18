"""Phase 1.3: Modality-specific SAE training and comparison.

Run as: ./env/bin/python scripts/run_phase1_saes.py
Logs to: results/phase1/modality_saes/sae_log.txt

Pipeline:
1. Extract ESM-3 activations at layers 16, 33, 42 under S-only and S+St conditions
2. Train TopK SAEs for each condition × layer (6 SAEs)
3. Annotate features and compare across conditions
4. Compare with ESM-2 Phase 0 SAE features

Target layers (from Experiment 1.2 CKA analysis):
  Layer 16: pre-integration (CKA(S,S+St)=0.16, modalities in separate spaces)
  Layer 33: integration midpoint (CKA crosses 0.5)
  Layer 42: post-integration (CKA=0.87, representations converging)
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

log_dir = ROOT / "results" / "phase1" / "modality_saes"
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "sae_log.txt"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("phase1_saes")

import torch
import numpy as np
import h5py

TARGET_LAYERS = [16, 33, 42]
CONDITIONS = ["S", "S+St"]


# ============================================================
# Step 1: Extract activations under each condition
# ============================================================
def step1_extract_activations():
    log.info("=" * 60)
    log.info("STEP 1: Extracting ESM-3 activations at target layers")
    log.info("=" * 60)

    from models.esm3_hooks import load_esm3, ESM3HookManager, tokenize_sequence
    from esm.sdk.api import ESMProtein

    # Load sequences
    with open(ROOT / "data" / "pilot" / "sequences" / "sequences.json") as f:
        seq_dict = json.load(f)
    with open(ROOT / "data" / "pilot" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    # Find proteins with downloaded AlphaFold structures
    struct_dir = ROOT / "data" / "pilot" / "structures"
    struct_files = {f.stem: str(f) for f in struct_dir.glob("*.pdb")} if struct_dir.exists() else {}
    log.info(f"Proteins with structures: {len(struct_files)}")

    # Load model
    device = "cuda:0"
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=device)
    log.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    hook_mgr = ESM3HookManager(model, layers=TARGET_LAYERS, extract_attention=False)
    max_length = 600

    act_dir = ROOT / "data" / "activations" / "esm3_multimodal"
    act_dir.mkdir(parents=True, exist_ok=True)

    # --- Condition S: sequence only (all 938 proteins) ---
    log.info("--- Extracting S-only activations ---")
    s_activations = {l: [] for l in TARGET_LAYERS}
    s_accessions = []
    s_lengths = []

    start_time = time.time()
    for i, (acc, seq) in enumerate(seq_dict.items()):
        if len(seq) > max_length:
            continue

        inputs = tokenize_sequence(seq, tokenizers, device=device)

        hook_mgr.cache.clear()
        hook_mgr.register()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**inputs)

        seq_len = len(seq)
        s_accessions.append(acc)
        s_lengths.append(seq_len)

        for layer_idx in TARGET_LAYERS:
            act = hook_mgr.cache.residual_stream[layer_idx]
            if act.ndim == 3:
                act = act[0]
            act = act[1:seq_len + 1].float().cpu()
            s_activations[layer_idx].append(act)

        hook_mgr.remove()

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            log.info(f"  S-only: {i+1}/{len(seq_dict)} ({(i+1)/elapsed:.1f} seq/s)")

    # Save S-only activations
    s_index = []
    offset = 0
    for acc, length in zip(s_accessions, s_lengths):
        s_index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length

    for layer_idx in TARGET_LAYERS:
        layer_acts = torch.cat(s_activations[layer_idx], dim=0)
        h5_path = act_dir / f"S_layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(), compression="gzip", compression_opts=4)
            f.attrs["condition"] = "S"
            f.attrs["layer"] = layer_idx
            f.attrs["num_proteins"] = len(s_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
        log.info(f"  Saved S layer {layer_idx}: {layer_acts.shape}")

    with open(act_dir / "S_residue_index.json", "w") as f:
        json.dump(s_index, f, indent=2)

    del s_activations
    torch.cuda.empty_cache()

    s_elapsed = time.time() - start_time
    log.info(f"S-only extraction: {len(s_accessions)} proteins, {offset} residues in {s_elapsed:.0f}s")

    # --- Condition S+St: sequence + structure ---
    log.info("--- Extracting S+St activations ---")
    sst_activations = {l: [] for l in TARGET_LAYERS}
    sst_accessions = []
    sst_lengths = []
    failed = 0

    start_time = time.time()
    for i, (acc, pdb_path) in enumerate(struct_files.items()):
        seq = seq_dict.get(acc)
        if seq is None or len(seq) > max_length:
            continue

        try:
            protein = ESMProtein.from_pdb(pdb_path)
            if protein.sequence is None:
                continue

            pdb_seq = protein.sequence
            tensor = model.encode(protein)

            kwargs = {"sequence_tokens": tensor.sequence.unsqueeze(0).to(device)}
            if tensor.structure is not None:
                kwargs["structure_tokens"] = tensor.structure.unsqueeze(0).to(device)

            hook_mgr.cache.clear()
            hook_mgr.register()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**kwargs)

            seq_len = len(pdb_seq)
            sst_accessions.append(acc)
            sst_lengths.append(seq_len)

            for layer_idx in TARGET_LAYERS:
                act = hook_mgr.cache.residual_stream[layer_idx]
                if act.ndim == 3:
                    act = act[0]
                act = act[1:seq_len + 1].float().cpu()
                sst_activations[layer_idx].append(act)

            hook_mgr.remove()

        except Exception as e:
            failed += 1
            if failed <= 3:
                log.warning(f"  {acc}: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            log.info(f"  S+St: {i+1}/{len(struct_files)} ({len(sst_accessions)} OK, {failed} failed)")

    # Save S+St activations
    sst_index = []
    offset = 0
    for acc, length in zip(sst_accessions, sst_lengths):
        sst_index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length

    for layer_idx in TARGET_LAYERS:
        layer_acts = torch.cat(sst_activations[layer_idx], dim=0)
        h5_path = act_dir / f"S+St_layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(), compression="gzip", compression_opts=4)
            f.attrs["condition"] = "S+St"
            f.attrs["layer"] = layer_idx
            f.attrs["num_proteins"] = len(sst_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
        log.info(f"  Saved S+St layer {layer_idx}: {layer_acts.shape}")

    with open(act_dir / "S+St_residue_index.json", "w") as f:
        json.dump(sst_index, f, indent=2)

    del sst_activations
    del model
    torch.cuda.empty_cache()

    sst_elapsed = time.time() - start_time
    log.info(f"S+St extraction: {len(sst_accessions)} proteins, {offset} residues in {sst_elapsed:.0f}s")

    return s_accessions, sst_accessions


# ============================================================
# Step 2: Train SAEs
# ============================================================
def step2_train_saes():
    log.info("=" * 60)
    log.info("STEP 2: Training modality-specific SAEs")
    log.info("=" * 60)

    from sae.model import SAEConfig
    from sae.train import SAETrainer, TrainConfig

    act_dir = ROOT / "data" / "activations" / "esm3_multimodal"
    model_dir = ROOT / "models" / "sae" / "esm3"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0"
    trainers = {}

    for condition in CONDITIONS:
        for layer_idx in TARGET_LAYERS:
            h5_path = act_dir / f"{condition}_layer_{layer_idx}.h5"
            with h5py.File(h5_path, "r") as f:
                activations = torch.tensor(f["activations"][:], dtype=torch.float32)

            log.info(f"\n--- Training SAE: {condition} layer {layer_idx} ---")
            log.info(f"  Activations: {activations.shape}")

            # Train/val split
            n = len(activations)
            perm = torch.randperm(n)
            n_train = int(0.9 * n)
            train_acts = activations[perm[:n_train]]
            val_acts = activations[perm[n_train:]]

            hidden_dim = activations.shape[1]  # 1536

            sae_config = SAEConfig(
                input_dim=hidden_dim,
                expansion_factor=8,  # 12,288 features
                k=64,
                architecture="topk",
            )

            safe_cond = condition.replace("+", "_")
            output_path = model_dir / f"{safe_cond}_layer_{layer_idx}_topk"
            output_path.mkdir(parents=True, exist_ok=True)

            train_config = TrainConfig(
                lr=3e-4,
                batch_size=4096,
                num_epochs=15,
                warmup_steps=500,
                log_every=200,
                device=device,
                output_dir=str(output_path),
            )

            trainer = SAETrainer(sae_config, train_config)
            log.info(f"  SAE: {sae_config.dict_size} features, K={sae_config.k}")
            log.info(f"  Training {n_train} samples, {n - n_train} validation...")

            summary = trainer.train_on_tensor(train_acts, val_acts)
            trainer.save(output_path / "final.pt")

            log.info(
                f"  Done: recon={summary['final_recon']:.4f}, "
                f"L0={summary['final_l0']:.1f}, "
                f"dead={summary['dead_frac']:.3f}"
            )

            trainers[(condition, layer_idx)] = trainer

            # Save training summary
            with open(output_path / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

    log.info("\nStep 2 complete: all 6 SAEs trained.")
    return trainers


# ============================================================
# Step 3: Annotate features
# ============================================================
def step3_annotate_features(trainers: dict):
    log.info("=" * 60)
    log.info("STEP 3: Annotating SAE features")
    log.info("=" * 60)

    act_dir = ROOT / "data" / "activations" / "esm3_multimodal"

    with open(ROOT / "data" / "pilot" / "sequences" / "sequences.json") as f:
        seq_dict = json.load(f)
    with open(ROOT / "data" / "pilot" / "annotations" / "metadata.json") as f:
        metadata = json.load(f)

    aa_set = list("ACDEFGHIKLMNPQRSTVWY")
    all_annotations = {}

    for (condition, layer_idx), trainer in trainers.items():
        log.info(f"\n--- Annotating: {condition} layer {layer_idx} ---")

        # Load activations and index
        h5_path = act_dir / f"{condition}_layer_{layer_idx}.h5"
        safe_cond = condition.replace("+", "_")
        index_path = act_dir / f"{condition}_residue_index.json"

        with h5py.File(h5_path, "r") as f:
            activations = torch.tensor(f["activations"][:], dtype=torch.float32)
        with open(index_path) as f:
            residue_index = json.load(f)

        # Build per-residue labels
        aa_labels = []
        functional_labels = []
        for entry in residue_index:
            acc = entry["accession"]
            seq = seq_dict.get(acc, "")
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

        # Encode through SAE
        device = trainer.train_config.device
        trainer.sae.eval()

        all_features = []
        batch_size = 8192
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i + batch_size].to(device)
            with torch.no_grad():
                z = trainer.sae.encode(batch)
            all_features.append(z.cpu())
        all_features = torch.cat(all_features, dim=0)

        # Feature stats
        feature_freq = (all_features > 0).float().mean(dim=0)
        active_count = (feature_freq > 0.001).sum().item()
        dead_count = (feature_freq == 0).sum().item()
        log.info(f"  Active features: {active_count}, Dead: {dead_count}")

        # Annotate top 200 features
        top_indices = feature_freq.argsort(descending=True)[:200]
        annotations = []

        bg_counts = {aa: (aa_labels == aa).sum() for aa in aa_set}
        total_bg = len(aa_labels)
        bg_fracs = {aa: count / total_bg for aa, count in bg_counts.items()}
        func_bg = functional_labels.mean()

        for feat_idx in top_indices:
            feat_idx = feat_idx.item()
            feat_acts = all_features[:, feat_idx]
            active_mask = feat_acts > 0

            if active_mask.sum() < 10:
                continue

            active_aas = aa_labels[active_mask.numpy()]
            total_active = len(active_aas)

            enrichments = {}
            for aa in aa_set:
                aa_frac = (active_aas == aa).sum() / total_active
                if bg_fracs[aa] > 0:
                    enrichments[aa] = aa_frac / bg_fracs[aa]

            top_enriched = sorted(enrichments.items(), key=lambda x: -x[1])[:3]

            func_in_active = functional_labels[active_mask.numpy()].mean()
            func_enrichment = func_in_active / func_bg if func_bg > 0 else 0

            annotations.append({
                "feature_idx": feat_idx,
                "activation_freq": feature_freq[feat_idx].item(),
                "mean_activation": feat_acts[active_mask].mean().item(),
                "top_enriched_aa": [(aa, round(e, 2)) for aa, e in top_enriched],
                "functional_site_enrichment": round(func_enrichment, 2),
                "functional_site_frac": round(float(func_in_active), 4),
            })

        key = f"{safe_cond}_layer_{layer_idx}"
        all_annotations[key] = annotations

        # Summary stats
        strong_aa = sum(1 for a in annotations if any(e > 3.0 for _, e in a["top_enriched_aa"]))
        func_enriched = sum(1 for a in annotations if a["functional_site_enrichment"] > 2.0)
        log.info(f"  Annotated: {len(annotations)} features")
        log.info(f"  Strong AA enrichment (>3x): {strong_aa}")
        log.info(f"  Functional site enrichment (>2x): {func_enriched}")

        # Save per-SAE annotations
        out_path = log_dir / f"{key}_annotations.json"
        with open(out_path, "w") as f:
            json.dump(annotations, f, indent=2)

    return all_annotations


# ============================================================
# Step 4: Cross-condition feature comparison
# ============================================================
def step4_compare_features(all_annotations: dict):
    log.info("=" * 60)
    log.info("STEP 4: Comparing features across modality conditions")
    log.info("=" * 60)

    comparison = {}

    for layer_idx in TARGET_LAYERS:
        s_key = f"S_layer_{layer_idx}"
        sst_key = f"S_St_layer_{layer_idx}"

        s_anns = all_annotations.get(s_key, [])
        sst_anns = all_annotations.get(sst_key, [])

        if not s_anns or not sst_anns:
            log.warning(f"  Missing annotations for layer {layer_idx}")
            continue

        log.info(f"\n--- Layer {layer_idx} comparison ---")

        # Compare AA enrichment profiles
        def get_aa_profile(anns):
            """Get the top AA enrichments across all features."""
            aa_features = {}
            for a in anns:
                for aa, enrich in a["top_enriched_aa"]:
                    if enrich > 3.0:
                        if aa not in aa_features:
                            aa_features[aa] = []
                        aa_features[aa].append(enrich)
            return {aa: (len(vals), max(vals)) for aa, vals in aa_features.items()}

        s_profile = get_aa_profile(s_anns)
        sst_profile = get_aa_profile(sst_anns)

        log.info(f"  S-only AA features (>3x enrichment):")
        for aa in sorted(s_profile.keys()):
            count, max_e = s_profile[aa]
            log.info(f"    {aa}: {count} features (max {max_e:.1f}x)")

        log.info(f"  S+St AA features (>3x enrichment):")
        for aa in sorted(sst_profile.keys()):
            count, max_e = sst_profile[aa]
            log.info(f"    {aa}: {count} features (max {max_e:.1f}x)")

        # Compare functional site enrichment
        s_func = [a for a in s_anns if a["functional_site_enrichment"] > 2.0]
        sst_func = [a for a in sst_anns if a["functional_site_enrichment"] > 2.0]

        log.info(f"  Functional site features: S-only={len(s_func)}, S+St={len(sst_func)}")

        if sst_func:
            log.info(f"  Top S+St functional features:")
            for a in sorted(sst_func, key=lambda x: -x["functional_site_enrichment"])[:5]:
                aa_str = ", ".join(f"{aa}:{e:.1f}x" for aa, e in a["top_enriched_aa"][:2])
                log.info(f"    Feature {a['feature_idx']}: func={a['functional_site_enrichment']:.1f}x, AA=[{aa_str}]")

        # Structure impact: does S+St have more functional features?
        s_mean_func = np.mean([a["functional_site_enrichment"] for a in s_anns]) if s_anns else 0
        sst_mean_func = np.mean([a["functional_site_enrichment"] for a in sst_anns]) if sst_anns else 0

        comparison[layer_idx] = {
            "s_only": {
                "total_annotated": len(s_anns),
                "strong_aa_features": sum(1 for a in s_anns if any(e > 3.0 for _, e in a["top_enriched_aa"])),
                "functional_features_2x": len(s_func),
                "mean_functional_enrichment": round(s_mean_func, 3),
                "aa_profile": {aa: {"count": c, "max_enrichment": m} for aa, (c, m) in s_profile.items()},
            },
            "s_plus_st": {
                "total_annotated": len(sst_anns),
                "strong_aa_features": sum(1 for a in sst_anns if any(e > 3.0 for _, e in a["top_enriched_aa"])),
                "functional_features_2x": len(sst_func),
                "mean_functional_enrichment": round(sst_mean_func, 3),
                "aa_profile": {aa: {"count": c, "max_enrichment": m} for aa, (c, m) in sst_profile.items()},
            },
        }

        log.info(f"  Mean functional enrichment: S-only={s_mean_func:.3f}, S+St={sst_mean_func:.3f}")

    # Save comparison
    with open(log_dir / "cross_condition_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    # Also compare with ESM-2 Phase 0 SAE
    log.info("\n--- Comparison with ESM-2 Phase 0 SAE (layer 24) ---")
    esm2_path = ROOT / "results" / "phase0" / "sae_reproduction" / "feature_annotations.json"
    if esm2_path.exists():
        with open(esm2_path) as f:
            esm2_anns = json.load(f)

        esm2_aa = sum(1 for a in esm2_anns if any(e > 3.0 for _, e in a["top_enriched_aa"]))
        esm2_func = sum(1 for a in esm2_anns if a["functional_site_enrichment"] > 2.0)
        log.info(f"  ESM-2 layer 24: {len(esm2_anns)} features, {esm2_aa} AA>3x, {esm2_func} func>2x")

        for layer_idx in TARGET_LAYERS:
            if layer_idx in comparison:
                c = comparison[layer_idx]
                log.info(
                    f"  ESM-3 S-only layer {layer_idx}: {c['s_only']['strong_aa_features']} AA>3x, "
                    f"{c['s_only']['functional_features_2x']} func>2x"
                )
                log.info(
                    f"  ESM-3 S+St  layer {layer_idx}: {c['s_plus_st']['strong_aa_features']} AA>3x, "
                    f"{c['s_plus_st']['functional_features_2x']} func>2x"
                )

    log.info("\nStep 4 complete.")
    return comparison


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("PHASE 1.3: Modality-Specific SAE Training")
    log.info(f"Target layers: {TARGET_LAYERS}")
    log.info(f"Conditions: {CONDITIONS}")
    log.info("=" * 60)

    overall_start = time.time()

    try:
        step1_extract_activations()
        trainers = step2_train_saes()
        all_annotations = step3_annotate_features(trainers)
        step4_compare_features(all_annotations)

        elapsed = time.time() - overall_start
        log.info(f"\nPHASE 1.3 COMPLETE in {elapsed/60:.1f} minutes")
        log.info(f"Results saved to {log_dir}/")

    except Exception as e:
        log.error(f"Phase 1.3 failed: {e}", exc_info=True)
        raise
