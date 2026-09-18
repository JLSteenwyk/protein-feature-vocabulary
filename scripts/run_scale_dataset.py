#!/usr/bin/env python3
"""Scale dataset from 199 to ~5,000 proteins with AlphaFold structures.

Run as: ./env/bin/python scripts/run_scale_dataset.py [--step N]

Staged pipeline — each step can be run independently:
  Step 1: Curate 2,500 proteins from UniProt
  Step 2: Download AlphaFold structures
  Step 3: Extract S-only activations (9 layers)
  Step 4: Extract S+St activations (3 layers + matching S-only)
  Step 5: Run DSSP on all structures
  Step 6: Re-run CKA modality dropout
  Step 7: Re-run attention atlas
  Step 8: Re-run functional attention
  Step 9: Train 6 SAEs (S/S+St × layers 16/33/42)
  Step 10: SS probing (S vs S+St)
  Step 11: SAE annotation + DSSP structural
  Step 12: Co-activation analysis
  Step 13: Decoder geometry analysis

All outputs go to data/scaled/ and results/phase1_scaled/.
Original pilot data at data/pilot/ and results/pilot_results/ are untouched.
"""

import sys
import os
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

# Directories
DATA_DIR = ROOT / "data" / "scaled"
STRUCT_DIR = DATA_DIR / "structures"
ANNOT_DIR = DATA_DIR / "annotations"
SEQ_DIR = DATA_DIR / "sequences"

ACT_S_DIR = ROOT / "data" / "activations" / "esm3_scaled"
ACT_MM_DIR = ROOT / "data" / "activations" / "esm3_scaled_multimodal"
MODEL_DIR = ROOT / "models" / "sae" / "esm3_scaled"
RESULTS_DIR = ROOT / "results" / "phase1_scaled"

MKDSSP = ROOT / "env" / "bin" / "mkdssp"

for d in [DATA_DIR, STRUCT_DIR, ANNOT_DIR, SEQ_DIR, ACT_S_DIR, ACT_MM_DIR,
          MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Logging
log_file = RESULTS_DIR / "scale_pipeline_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scale")

# Constants
LAYERS_9 = [0, 6, 12, 18, 24, 30, 36, 42, 47]
TARGET_LAYERS = [16, 33, 42]
MAX_SEQ_LEN = 600


# ============================================================
# Step 1: Curate 2,500 proteins from UniProt
# ============================================================
def step1_curate_proteins():
    """Query UniProt REST API for ~2,500 diverse reviewed proteins."""
    log.info("=" * 60)
    log.info("STEP 1: Curating ~5,000 proteins from UniProt")
    log.info("=" * 60)

    seq_path = SEQ_DIR / "sequences.json"
    meta_path = ANNOT_DIR / "metadata.json"
    if seq_path.exists() and meta_path.exists():
        with open(seq_path) as f:
            seq_dict = json.load(f)
        with open(meta_path) as f:
            metadata = json.load(f)
        log.info(f"Step 1: Already have {len(seq_dict)} proteins, skipping download")
        return seq_dict, metadata

    from data.download import search_uniprot, _extract_protein_name, _extract_go_terms
    from data.download import _extract_ec, _extract_keywords, _extract_pdb_ids
    from data.download import _extract_xrefs, _extract_features

    # Scaled-up queries (~5× the pilot targets, with broader categories)
    queries = [
        {
            "name": "enzymes_short",
            "query": "(reviewed:true) AND (ec:*) AND (length:[50 TO 200]) AND (structure_3d:true)",
            "target_count": 500,
        },
        {
            "name": "enzymes_medium",
            "query": "(reviewed:true) AND (ec:*) AND (length:[200 TO 400]) AND (structure_3d:true)",
            "target_count": 500,
        },
        {
            "name": "enzymes_long",
            "query": "(reviewed:true) AND (ec:*) AND (length:[400 TO 600]) AND (structure_3d:true)",
            "target_count": 400,
        },
        {
            "name": "binding_proteins",
            "query": "(reviewed:true) AND (keyword:KW-0067) AND (length:[50 TO 500]) AND (structure_3d:true)",
            "target_count": 500,
        },
        {
            "name": "transporters",
            "query": "(reviewed:true) AND (keyword:KW-0813) AND (length:[100 TO 600]) AND (structure_3d:true)",
            "target_count": 400,
        },
        {
            "name": "dna_binding",
            "query": "(reviewed:true) AND (keyword:KW-0238) AND (length:[50 TO 400]) AND (structure_3d:true)",
            "target_count": 400,
        },
        {
            "name": "structural_proteins",
            "query": "(reviewed:true) AND (keyword:KW-0732) AND (length:[50 TO 500]) AND (structure_3d:true)",
            "target_count": 250,
        },
        {
            "name": "receptors",
            "query": "(reviewed:true) AND (keyword:KW-0675) AND (length:[100 TO 600]) AND (structure_3d:true)",
            "target_count": 400,
        },
        {
            "name": "small_proteins",
            "query": "(reviewed:true) AND (length:[50 TO 100]) AND (structure_3d:true)",
            "target_count": 400,
        },
        {
            "name": "signaling",
            "query": "(reviewed:true) AND (keyword:KW-0807) AND (length:[100 TO 500]) AND (structure_3d:true)",
            "target_count": 400,
        },
        {
            "name": "immune",
            "query": "(reviewed:true) AND (keyword:KW-0391) AND (length:[50 TO 500]) AND (structure_3d:true)",
            "target_count": 250,
        },
        {
            "name": "diverse_annotated",
            "query": "(reviewed:true) AND (length:[100 TO 500]) AND (structure_3d:true) AND (annotation_score:5)",
            "target_count": 1000,
        },
    ]

    fields = [
        "accession", "id", "protein_name", "gene_names", "organism_name",
        "length", "sequence", "go_id", "go_p", "go_f", "ec", "keyword",
        "ft_act_site", "ft_binding", "ft_site", "xref_pdb", "xref_interpro", "xref_pfam",
    ]

    all_proteins = {}
    seen_accessions = set()

    for q in queries:
        log.info(f"Querying: {q['name']} (target: {q['target_count']})...")
        try:
            results = search_uniprot(q["query"], fields, size=q["target_count"] * 2)
        except Exception as e:
            log.warning(f"  Query failed: {e}")
            continue

        count = 0
        for protein in results:
            acc = protein.get("primaryAccession", "")
            if acc in seen_accessions:
                continue
            seen_accessions.add(acc)

            seq = protein.get("sequence", {}).get("value", "")
            if not seq or len(seq) < 30 or len(seq) > MAX_SEQ_LEN:
                continue

            all_proteins[acc] = {
                "accession": acc,
                "entry_name": protein.get("uniProtkbId", ""),
                "protein_name": _extract_protein_name(protein),
                "organism": protein.get("organism", {}).get("scientificName", ""),
                "length": len(seq),
                "sequence": seq,
                "category": q["name"],
                "go_terms": _extract_go_terms(protein),
                "ec_numbers": _extract_ec(protein),
                "keywords": _extract_keywords(protein),
                "pdb_ids": _extract_pdb_ids(protein),
                "interpro": _extract_xrefs(protein, "InterPro"),
                "pfam": _extract_xrefs(protein, "Pfam"),
                "features": _extract_features(protein),
            }
            count += 1
            if count >= q["target_count"]:
                break

        log.info(f"  Got {count} proteins (total so far: {len(all_proteins)})")
        import time as _t
        _t.sleep(1)

    # Save sequences
    seq_dict = {acc: p["sequence"] for acc, p in all_proteins.items()}
    with open(seq_path, "w") as f:
        json.dump(seq_dict, f)

    # Save FASTA
    fasta_path = SEQ_DIR / "sequences.fasta"
    with open(fasta_path, "w") as f:
        for acc, p in all_proteins.items():
            f.write(f">{acc} {p['entry_name']} {p['protein_name']}\n")
            seq = p["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    # Save metadata (without sequences to save space)
    metadata = {}
    for acc, p in all_proteins.items():
        metadata[acc] = {k: v for k, v in p.items() if k != "sequence"}
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    cats = defaultdict(int)
    for p in all_proteins.values():
        cats[p["category"]] += 1
    lengths = [p["length"] for p in all_proteins.values()]

    summary = {
        "total_proteins": len(all_proteins),
        "categories": dict(cats),
        "length_stats": {
            "min": min(lengths), "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
        },
        "with_pdb": sum(1 for p in all_proteins.values() if p["pdb_ids"]),
        "with_features": sum(1 for p in all_proteins.values() if p["features"]),
        "with_interpro": sum(1 for p in all_proteins.values() if p["interpro"]),
    }
    with open(DATA_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"\nStep 1 complete:")
    log.info(f"  Total proteins: {summary['total_proteins']}")
    log.info(f"  With PDB: {summary['with_pdb']}")
    log.info(f"  With functional annotations: {summary['with_features']}")
    log.info(f"  Length range: {summary['length_stats']['min']}-{summary['length_stats']['max']} "
             f"(mean: {summary['length_stats']['mean']:.0f})")
    for cat, count in sorted(cats.items()):
        log.info(f"    {cat}: {count}")

    return seq_dict, metadata


# ============================================================
# Step 2: Download AlphaFold structures
# ============================================================
def step2_download_structures(seq_dict: dict):
    """Download AlphaFold structures for all curated proteins."""
    log.info("=" * 60)
    log.info("STEP 2: Downloading AlphaFold structures")
    log.info("=" * 60)

    import urllib.request

    accs = list(seq_dict.keys())
    available = {}
    failed = []

    # Check existing
    for acc in accs:
        pdb_path = STRUCT_DIR / f"{acc}.pdb"
        if pdb_path.exists():
            available[acc] = str(pdb_path)

    if available:
        log.info(f"  Already have {len(available)} structures, downloading remaining...")

    start_time = time.time()
    remaining = [acc for acc in accs if acc not in available]

    for i, acc in enumerate(remaining):
        pdb_path = STRUCT_DIR / f"{acc}.pdb"
        try:
            api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{acc}"
            resp = urllib.request.urlopen(api_url, timeout=15)
            data = json.loads(resp.read())
            if data and data[0].get("pdbUrl"):
                pdb_url = data[0]["pdbUrl"]
                urllib.request.urlretrieve(pdb_url, pdb_path)
                available[acc] = str(pdb_path)
            else:
                failed.append(acc)
        except Exception as e:
            failed.append(acc)
            if len(failed) <= 10:
                log.warning(f"  Failed {acc}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            log.info(f"  Downloaded {i+1}/{len(remaining)} "
                     f"({len(available)} OK, {len(failed)} failed, {rate:.1f}/s)")

        # Brief pause every 50 to be polite to the API
        if (i + 1) % 50 == 0:
            time.sleep(0.5)

    elapsed = time.time() - start_time
    log.info(f"\nStep 2 complete in {elapsed:.0f}s:")
    log.info(f"  Structures available: {len(available)}")
    log.info(f"  Failed downloads: {len(failed)}")

    # Save structure map
    with open(DATA_DIR / "structure_map.json", "w") as f:
        json.dump({"available": list(available.keys()), "failed": failed}, f, indent=2)

    return available


# ============================================================
# Step 3: Extract S-only activations (9 layers)
# ============================================================
def step3_extract_s_only(seq_dict: dict):
    """Extract sequence-only ESM-3 activations at 9 layers for all proteins."""
    log.info("=" * 60)
    log.info("STEP 3: Extracting S-only activations (9 layers)")
    log.info("=" * 60)

    import torch
    import h5py

    # Check if already done
    index_path = ACT_S_DIR / "residue_index.json"
    if index_path.exists():
        with open(index_path) as f:
            existing_index = json.load(f)
        if len(existing_index) >= len(seq_dict) * 0.9:
            log.info(f"Step 3: Already have {len(existing_index)} proteins extracted, skipping")
            return

    from models.esm3_hooks import load_esm3, ESM3HookManager, tokenize_sequence

    device = "cuda:0"
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=device)
    log.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    hook_mgr = ESM3HookManager(model, layers=LAYERS_9, extract_attention=False)

    all_activations = {l: [] for l in LAYERS_9}
    all_accessions = []
    all_lengths = []
    skipped = 0

    start_time = time.time()
    accs = list(seq_dict.keys())

    for i, acc in enumerate(accs):
        seq = seq_dict[acc]
        if len(seq) > MAX_SEQ_LEN:
            skipped += 1
            continue

        inputs = tokenize_sequence(seq, tokenizers, device=device)

        hook_mgr.cache.clear()
        hook_mgr.register()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(**inputs)

        seq_len = len(seq)
        all_accessions.append(acc)
        all_lengths.append(seq_len)

        for layer_idx in LAYERS_9:
            act = hook_mgr.cache.residual_stream[layer_idx]
            if act.ndim == 3:
                act = act[0]
            act = act[1:seq_len + 1].float().cpu()
            all_activations[layer_idx].append(act)

        hook_mgr.remove()

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1 - skipped) / elapsed
            vram = torch.cuda.memory_allocated(0) / 1e9
            log.info(f"  S-only: {i+1}/{len(accs)} ({rate:.1f} seq/s, VRAM: {vram:.1f} GB)")

    # Save to HDF5
    for layer_idx in LAYERS_9:
        layer_acts = torch.cat(all_activations[layer_idx], dim=0)
        h5_path = ACT_S_DIR / f"layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(),
                             compression="gzip", compression_opts=4)
            f.attrs["layer"] = layer_idx
            f.attrs["model"] = "esm3_sm_open_v1"
            f.attrs["condition"] = "S"
            f.attrs["num_proteins"] = len(all_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
            f.attrs["hidden_dim"] = layer_acts.shape[1]
        log.info(f"  Saved S layer {layer_idx}: {layer_acts.shape}")
        del layer_acts

    # Save residue index
    index = []
    offset = 0
    for acc, length in zip(all_accessions, all_lengths):
        index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    del all_activations
    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    log.info(f"\nStep 3 complete in {elapsed/60:.1f} min:")
    log.info(f"  Proteins: {len(all_accessions)} (skipped {skipped})")
    log.info(f"  Total residues: {offset}")


# ============================================================
# Step 4: Extract S+St activations (3 layers)
# ============================================================
def step4_extract_multimodal(seq_dict: dict, available_structures: dict):
    """Extract S+St and matching S-only activations at target layers."""
    log.info("=" * 60)
    log.info("STEP 4: Extracting S+St activations (layers 16, 33, 42)")
    log.info("=" * 60)

    import torch
    import h5py
    from esm.sdk.api import ESMProtein

    # Check if already done
    sst_index_path = ACT_MM_DIR / "S+St_residue_index.json"
    if sst_index_path.exists():
        with open(sst_index_path) as f:
            existing_index = json.load(f)
        if len(existing_index) >= len(available_structures) * 0.8:
            log.info(f"Step 4: Already have {len(existing_index)} proteins, skipping")
            return

    from models.esm3_hooks import load_esm3, ESM3HookManager, tokenize_sequence

    device = "cuda:0"
    log.info("Loading ESM-3...")
    model, tokenizers = load_esm3(device=device)
    log.info(f"Model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    hook_mgr = ESM3HookManager(model, layers=TARGET_LAYERS, extract_attention=False)

    # --- S-only at target layers (for matched comparison) ---
    log.info("--- Extracting S-only at target layers ---")
    s_activations = {l: [] for l in TARGET_LAYERS}
    s_accessions = []
    s_lengths = []

    start_time = time.time()
    struct_accs = list(available_structures.keys())

    for i, acc in enumerate(struct_accs):
        seq = seq_dict.get(acc)
        if seq is None or len(seq) > MAX_SEQ_LEN:
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

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            log.info(f"  S-only (matched): {i+1}/{len(struct_accs)} ({len(s_accessions)} OK)")

    # Save S-only at target layers
    s_index = []
    offset = 0
    for acc, length in zip(s_accessions, s_lengths):
        s_index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length

    for layer_idx in TARGET_LAYERS:
        layer_acts = torch.cat(s_activations[layer_idx], dim=0)
        h5_path = ACT_MM_DIR / f"S_layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(),
                             compression="gzip", compression_opts=4)
            f.attrs["condition"] = "S"
            f.attrs["layer"] = layer_idx
            f.attrs["num_proteins"] = len(s_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
        log.info(f"  Saved S layer {layer_idx}: {layer_acts.shape}")
        del layer_acts

    with open(ACT_MM_DIR / "S_residue_index.json", "w") as f:
        json.dump(s_index, f, indent=2)

    del s_activations
    torch.cuda.empty_cache()

    s_elapsed = time.time() - start_time
    log.info(f"S-only (matched): {len(s_accessions)} proteins, {offset} residues in {s_elapsed:.0f}s")

    # --- S+St: sequence + structure ---
    log.info("--- Extracting S+St activations ---")
    sst_activations = {l: [] for l in TARGET_LAYERS}
    sst_accessions = []
    sst_lengths = []
    failed = 0

    start_time = time.time()
    for i, acc in enumerate(struct_accs):
        seq = seq_dict.get(acc)
        pdb_path = available_structures.get(acc)
        if seq is None or pdb_path is None or len(seq) > MAX_SEQ_LEN:
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
            if failed <= 10:
                log.warning(f"  {acc}: {e}")

        if (i + 1) % 200 == 0:
            elapsed = time.time() - start_time
            log.info(f"  S+St: {i+1}/{len(struct_accs)} ({len(sst_accessions)} OK, {failed} failed)")

    # Save S+St activations
    sst_index = []
    offset = 0
    for acc, length in zip(sst_accessions, sst_lengths):
        sst_index.append({"accession": acc, "start": offset, "end": offset + length, "length": length})
        offset += length

    for layer_idx in TARGET_LAYERS:
        layer_acts = torch.cat(sst_activations[layer_idx], dim=0)
        h5_path = ACT_MM_DIR / f"S+St_layer_{layer_idx}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("activations", data=layer_acts.numpy(),
                             compression="gzip", compression_opts=4)
            f.attrs["condition"] = "S+St"
            f.attrs["layer"] = layer_idx
            f.attrs["num_proteins"] = len(sst_accessions)
            f.attrs["num_residues"] = layer_acts.shape[0]
        log.info(f"  Saved S+St layer {layer_idx}: {layer_acts.shape}")
        del layer_acts

    with open(sst_index_path, "w") as f:
        json.dump(sst_index, f, indent=2)

    del sst_activations
    del model
    torch.cuda.empty_cache()

    sst_elapsed = time.time() - start_time
    log.info(f"\nStep 4 complete:")
    log.info(f"  S+St: {len(sst_accessions)} proteins, {offset} residues in {sst_elapsed:.0f}s")
    log.info(f"  Failed: {failed}")


# ============================================================
# Step 5: Run DSSP on all structures
# ============================================================
def step5_run_dssp(available_structures: dict):
    """Run mkdssp on all structures to extract SS and RSA."""
    log.info("=" * 60)
    log.info("STEP 5: Running DSSP on all structures")
    log.info("=" * 60)

    dssp_path = ANNOT_DIR / "dssp_annotations.json"
    if dssp_path.exists():
        with open(dssp_path) as f:
            existing = json.load(f)
        if len(existing) >= len(available_structures) * 0.8:
            log.info(f"Step 5: Already have DSSP for {len(existing)} proteins, skipping")
            return existing

    SS8_TO_SS3 = {
        "H": "H", "G": "H", "I": "H",
        "E": "E", "B": "E",
        "T": "C", "S": "C", " ": "C", "C": "C", "P": "C",
    }

    dssp_results = {}
    failed = 0
    start_time = time.time()

    for i, (acc, pdb_path) in enumerate(available_structures.items()):
        try:
            result = subprocess.run(
                [str(MKDSSP), "--output-format", "dssp", str(pdb_path)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                failed += 1
                continue

            # Parse DSSP output
            lines = result.stdout.strip().split("\n")
            in_residue = False
            residues = []
            for line in lines:
                if line.startswith("  #  RESIDUE"):
                    in_residue = True
                    continue
                if not in_residue or len(line) < 40:
                    continue

                resnum_str = line[5:10].strip()
                chain = line[11]
                aa1 = line[13]
                ss8 = line[16] if len(line) > 16 else " "
                ss3 = SS8_TO_SS3.get(ss8, "C")

                # ASA: columns 35-38
                try:
                    asa = float(line[35:38].strip())
                except (ValueError, IndexError):
                    asa = 0.0

                if aa1 not in "ACDEFGHIKLMNPQRSTVWY":
                    continue

                residues.append({
                    "resnum": int(resnum_str) if resnum_str.isdigit() else 0,
                    "aa": aa1,
                    "ss8": ss8.strip() or "C",
                    "ss3": ss3,
                    "asa": asa,
                })

            if residues:
                dssp_results[acc] = residues

        except Exception:
            failed += 1

        if (i + 1) % 500 == 0:
            log.info(f"  DSSP: {i+1}/{len(available_structures)} ({len(dssp_results)} OK, {failed} failed)")

    with open(dssp_path, "w") as f:
        json.dump(dssp_results, f)

    elapsed = time.time() - start_time
    log.info(f"\nStep 5 complete in {elapsed:.0f}s:")
    log.info(f"  DSSP annotations: {len(dssp_results)} proteins")
    log.info(f"  Failed: {failed}")

    return dssp_results


# ============================================================
# Step 6: Re-run CKA modality dropout
# ============================================================
def step6_cka_modality(seq_dict: dict, available_structures: dict, metadata: dict):
    """CKA analysis: S vs S+St vs S+F vs S+St+F at 13 layers."""
    log.info("=" * 60)
    log.info("STEP 6: CKA Modality Dropout (scaled)")
    log.info("=" * 60)

    import torch
    import numpy as np

    cka_dir = RESULTS_DIR / "modality_dropout"
    cka_dir.mkdir(parents=True, exist_ok=True)

    cka_path = cka_dir / "cka_results.json"
    if cka_path.exists():
        log.info("Step 6: CKA results already exist, skipping")
        return

    from models.esm3_hooks import load_esm3, ESM3HookManager
    from esm.sdk.api import ESMProtein
    from esm.utils.types import FunctionAnnotation

    # Select subset with structures and InterPro annotations
    subset_accs = []
    for acc in available_structures:
        if acc in seq_dict and len(seq_dict[acc]) <= 500:
            m = metadata.get(acc, {})
            if m.get("interpro"):
                subset_accs.append(acc)
        if len(subset_accs) >= 500:
            break

    # If not enough with InterPro, fill with structure-only
    if len(subset_accs) < 300:
        for acc in available_structures:
            if acc not in set(subset_accs) and acc in seq_dict and len(seq_dict[acc]) <= 500:
                subset_accs.append(acc)
            if len(subset_accs) >= 500:
                break

    log.info(f"Selected {len(subset_accs)} proteins for CKA")

    device = "cuda:0"
    model, tokenizers = load_esm3(device=device)

    layers = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]
    hook_mgr = ESM3HookManager(model, layers=layers, extract_attention=False)

    # Tokenize under all 4 conditions
    conditions = ["S", "S+St", "S+F", "S+St+F"]
    tokenized = {}
    successful = 0

    for acc in subset_accs:
        seq = seq_dict[acc]
        pdb_path = available_structures[acc]
        m = metadata.get(acc, {})
        interpro_ids = m.get("interpro", [])

        try:
            protein_with_struct = ESMProtein.from_pdb(pdb_path)
            if protein_with_struct.sequence is None:
                continue

            pdb_seq = protein_with_struct.sequence

            func_annotations = []
            for ipr_id in interpro_ids:
                func_annotations.append(
                    FunctionAnnotation(label=ipr_id, start=1, end=len(pdb_seq))
                )

            conds = {}
            conds["S"] = model.encode(ESMProtein(sequence=pdb_seq))
            conds["S+St"] = model.encode(protein_with_struct)

            protein_sf = ESMProtein(
                sequence=pdb_seq,
                function_annotations=func_annotations if func_annotations else None,
            )
            conds["S+F"] = model.encode(protein_sf)

            protein_full = ESMProtein(
                sequence=pdb_seq,
                coordinates=protein_with_struct.coordinates,
                function_annotations=func_annotations if func_annotations else None,
            )
            conds["S+St+F"] = model.encode(protein_full)

            tokenized[acc] = conds
            successful += 1

            if successful % 50 == 0:
                log.info(f"  Tokenized {successful} proteins")

        except Exception as e:
            if successful < 5:
                log.warning(f"  {acc}: {e}")

    log.info(f"Tokenized {successful} proteins under 4 conditions")

    # Extract activations
    activations = {cond: {l: [] for l in layers} for cond in conditions}

    start_time = time.time()
    processed = 0
    for acc, cond_tensors in tokenized.items():
        for cond_name in conditions:
            if cond_name not in cond_tensors:
                continue
            pt = cond_tensors[cond_name]

            kwargs = {}
            if pt.sequence is not None:
                kwargs["sequence_tokens"] = pt.sequence.unsqueeze(0).to(device)
            if pt.structure is not None:
                kwargs["structure_tokens"] = pt.structure.unsqueeze(0).to(device)
            if pt.function is not None:
                kwargs["function_tokens"] = pt.function.unsqueeze(0).to(device)
            if pt.residue_annotations is not None:
                kwargs["residue_annotation_tokens"] = pt.residue_annotations.unsqueeze(0).to(device)

            hook_mgr.cache.clear()
            hook_mgr.register()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**kwargs)

            seq_len = kwargs["sequence_tokens"].shape[1] - 2
            for layer_idx in layers:
                act = hook_mgr.cache.residual_stream[layer_idx]
                if act.ndim == 3:
                    act = act[0]
                act = act[1:seq_len + 1].float().cpu()
                activations[cond_name][layer_idx].append(act)

            hook_mgr.remove()

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            log.info(f"  Activations: {processed}/{len(tokenized)} ({processed/elapsed:.1f} prot/s)")

    # Concatenate
    for cond in conditions:
        for l in layers:
            if activations[cond][l]:
                activations[cond][l] = torch.cat(activations[cond][l], dim=0).numpy()
            else:
                activations[cond][l] = None

    del model
    torch.cuda.empty_cache()

    # Compute CKA
    def linear_cka(X, Y):
        X = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
        Y = Y.astype(np.float64) - Y.astype(np.float64).mean(axis=0, keepdims=True)
        hsic_xy = np.sum((X.T @ Y) ** 2)
        hsic_xx = np.sum((X.T @ X) ** 2)
        hsic_yy = np.sum((Y.T @ Y) ** 2)
        denom = np.sqrt(hsic_xx * hsic_yy)
        return float(hsic_xy / denom) if denom > 1e-12 else 0.0

    results = {}
    for layer_idx in layers:
        cka_matrix = np.zeros((len(conditions), len(conditions)))
        for i, ci in enumerate(conditions):
            for j, cj in enumerate(conditions):
                if i > j:
                    cka_matrix[i, j] = cka_matrix[j, i]
                    continue
                X = activations[ci][layer_idx]
                Y = activations[cj][layer_idx]
                if X is None or Y is None:
                    cka_matrix[i, j] = np.nan
                    continue
                n = min(X.shape[0], Y.shape[0], 50000)
                if X.shape[0] > n:
                    idx = np.random.choice(X.shape[0], n, replace=False)
                    X, Y = X[idx], Y[idx]
                cka_matrix[i, j] = linear_cka(X[:n], Y[:n])
        results[layer_idx] = cka_matrix.tolist()

        log.info(f"  Layer {layer_idx}: CKA(S,S+St)={cka_matrix[0,1]:.4f}, "
                 f"CKA(S,S+F)={cka_matrix[0,2]:.4f}")

    output = {
        "layers": layers,
        "conditions": conditions,
        "cka_matrices": {str(k): v for k, v in results.items()},
        "num_proteins": len(tokenized),
    }
    with open(cka_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Step 6 complete: CKA saved to {cka_path}")


# ============================================================
# Step 7: Re-run attention atlas
# ============================================================
def step7_attention_atlas(seq_dict: dict, available_structures: dict):
    """Attention atlas: JSD(S, S+St) for all 48 layers × 24 heads."""
    log.info("=" * 60)
    log.info("STEP 7: Attention Atlas (scaled)")
    log.info("=" * 60)

    import torch
    import torch.nn.functional as F
    import numpy as np
    import einops
    import functools

    atlas_dir = RESULTS_DIR / "attention_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    atlas_path = atlas_dir / "attention_atlas.json"
    if atlas_path.exists():
        log.info("Step 7: Attention atlas already exists, skipping")
        return

    from models.esm3_hooks import load_esm3
    from esm.sdk.api import ESMProtein

    # Select proteins (cap at 500 for attention — it's expensive)
    subset_accs = []
    for acc in available_structures:
        if acc in seq_dict and len(seq_dict[acc]) <= 400:
            subset_accs.append(acc)
        if len(subset_accs) >= 500:
            break
    log.info(f"Selected {len(subset_accs)} proteins for attention atlas")

    device = "cuda:0"
    model, tokenizers = load_esm3(device=device)
    all_layers = list(range(48))
    n_heads = 24

    # AttentionCapture: replicate full attention computation to extract weights
    class AttentionCapture:
        def __init__(self, model, layers):
            self.model = model
            self.layers = layers
            self.weights = {}
            self._orig_forwards = {}

        def install(self):
            for li in self.layers:
                attn = self.model.transformer.blocks[li].attn
                self._orig_forwards[li] = attn.forward
                attn.forward = self._make_capturing_forward(attn, li)

        def _make_capturing_forward(self, attn_mod, layer_idx):
            storage = self.weights
            def capturing_forward(x, seq_id):
                qkv_BLD3 = attn_mod.layernorm_qkv(x)
                query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
                query_BLD, key_BLD = (
                    attn_mod.q_ln(query_BLD).to(query_BLD.dtype),
                    attn_mod.k_ln(key_BLD).to(query_BLD.dtype),
                )
                query_BLD, key_BLD = attn_mod._apply_rotary(query_BLD, key_BLD)
                reshaper = functools.partial(
                    einops.rearrange, pattern="b s (h d) -> b h s d", h=attn_mod.n_heads
                )
                query_BHLD, key_BHLD, value_BHLD = map(
                    reshaper, (query_BLD, key_BLD, value_BLD)
                )
                d_head = attn_mod.d_head
                attn_logits = torch.matmul(
                    query_BHLD, key_BHLD.transpose(-2, -1)
                ) / (d_head ** 0.5)
                if seq_id is not None:
                    mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                    mask_BHLL = mask_BLL.unsqueeze(1)
                    attn_logits = attn_logits.masked_fill(~mask_BHLL, float("-inf"))
                else:
                    mask_BHLL = None
                attn_w = torch.softmax(attn_logits, dim=-1)
                storage[layer_idx] = attn_w.detach().float().cpu()
                if mask_BHLL is not None:
                    context_BHLD = F.scaled_dot_product_attention(
                        query_BHLD, key_BHLD, value_BHLD, mask_BHLL
                    )
                else:
                    context_BHLD = F.scaled_dot_product_attention(
                        query_BHLD, key_BHLD, value_BHLD
                    )
                context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
                return attn_mod.out_proj(context_BLD)
            return capturing_forward

        def remove(self):
            for li, orig in self._orig_forwards.items():
                self.model.transformer.blocks[li].attn.forward = orig
            self._orig_forwards.clear()
            self.weights.clear()

    capture = AttentionCapture(model, all_layers)

    # Per-protein: run S and S+St, compute JSD per head
    jsd_accumulator = np.zeros((48, n_heads))
    entropy_s = np.zeros((48, n_heads))
    entropy_sst = np.zeros((48, n_heads))
    count = 0

    def jsd(p, q, eps=1e-10):
        m = 0.5 * (p + q)
        kl_pm = (p * np.log((p + eps) / (m + eps))).sum(-1)
        kl_qm = (q * np.log((q + eps) / (m + eps))).sum(-1)
        return 0.5 * (kl_pm + kl_qm)

    def entropy(p, eps=1e-10):
        return -(p * np.log(p + eps)).sum(-1)

    start_time = time.time()

    for i, acc in enumerate(subset_accs):
        seq = seq_dict[acc]
        pdb_path = available_structures[acc]

        try:
            protein = ESMProtein.from_pdb(pdb_path)
            if protein.sequence is None:
                continue
            pdb_seq = protein.sequence

            # S-only
            from models.esm3_hooks import tokenize_sequence
            inputs_s = tokenize_sequence(pdb_seq, tokenizers, device=device)

            capture.weights.clear()
            capture.install()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**inputs_s)
            attn_s = {l: w.float().numpy() for l, w in capture.weights.items()}
            capture.remove()

            # S+St
            tensor = model.encode(protein)
            kwargs_sst = {"sequence_tokens": tensor.sequence.unsqueeze(0).to(device)}
            if tensor.structure is not None:
                kwargs_sst["structure_tokens"] = tensor.structure.unsqueeze(0).to(device)

            capture.weights.clear()
            capture.install()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**kwargs_sst)
            attn_sst = {l: w.float().numpy() for l, w in capture.weights.items()}
            capture.remove()

            # Compute JSD per head (average over query positions)
            for li in all_layers:
                if li not in attn_s or li not in attn_sst:
                    continue
                ws = attn_s[li][0]   # (n_heads, L, L)
                wst = attn_sst[li][0]
                L = min(ws.shape[1], wst.shape[1])
                ws = ws[:, :L, :L]
                wst = wst[:, :L, :L]

                for h in range(n_heads):
                    j = jsd(ws[h], wst[h]).mean()
                    jsd_accumulator[li, h] += j
                    entropy_s[li, h] += entropy(ws[h]).mean()
                    entropy_sst[li, h] += entropy(wst[h]).mean()

            count += 1

            if count % 50 == 0:
                elapsed = time.time() - start_time
                log.info(f"  Atlas: {count}/{len(subset_accs)} ({count/elapsed:.1f} prot/s)")

        except Exception as e:
            if count < 5:
                log.warning(f"  {acc}: {e}")

    if count == 0:
        log.error("No proteins processed for attention atlas")
        return

    jsd_mean = jsd_accumulator / count
    entropy_s_mean = entropy_s / count
    entropy_sst_mean = entropy_sst / count

    del model
    torch.cuda.empty_cache()

    output = {
        "num_proteins": count,
        "layers": all_layers,
        "n_heads": n_heads,
        "jsd_matrix": jsd_mean.tolist(),
        "entropy_s_matrix": entropy_s_mean.tolist(),
        "entropy_sst_matrix": entropy_sst_mean.tolist(),
    }
    with open(atlas_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    top_jsd = []
    for li in range(48):
        for h in range(n_heads):
            top_jsd.append((li, h, jsd_mean[li, h]))
    top_jsd.sort(key=lambda x: -x[2])

    log.info(f"\nStep 7 complete ({count} proteins):")
    log.info("Top 10 structure-responsive heads:")
    for li, h, j in top_jsd[:10]:
        log.info(f"  Layer {li} Head {h}: JSD = {j:.4f}")


# ============================================================
# Step 8: Re-run functional attention
# ============================================================
def step8_functional_attention(seq_dict: dict, available_structures: dict, metadata: dict):
    """Functional attention: enrichment at functional sites for key layers."""
    log.info("=" * 60)
    log.info("STEP 8: Functional Attention (scaled)")
    log.info("=" * 60)

    import torch
    import torch.nn.functional as F
    import numpy as np
    import einops
    import functools

    func_dir = RESULTS_DIR / "functional_attention"
    func_dir.mkdir(parents=True, exist_ok=True)

    func_path = func_dir / "functional_attention_results.json"
    if func_path.exists():
        log.info("Step 8: Functional attention results exist, skipping")
        return

    from models.esm3_hooks import load_esm3, tokenize_sequence
    from esm.sdk.api import ESMProtein

    # Select proteins WITH functional annotations
    subset_accs = []
    for acc in available_structures:
        if acc not in seq_dict or len(seq_dict[acc]) > 400:
            continue
        m = metadata.get(acc, {})
        feats = m.get("features", [])
        if feats:
            subset_accs.append(acc)
        if len(subset_accs) >= 500:
            break

    log.info(f"Selected {len(subset_accs)} proteins with functional annotations")

    device = "cuda:0"
    model, tokenizers = load_esm3(device=device)

    n_heads = 24
    target_layers = [0, 2, 4, 6, 8, 12, 16, 24, 32, 47]

    # AttentionCapture: replicate full attention computation to extract weights
    class AttentionCapture:
        def __init__(self, model, layers):
            self.model = model
            self.layers = layers
            self.weights = {}
            self._orig_forwards = {}

        def install(self):
            for li in self.layers:
                attn = self.model.transformer.blocks[li].attn
                self._orig_forwards[li] = attn.forward
                attn.forward = self._make_capturing_forward(attn, li)

        def _make_capturing_forward(self, attn_mod, layer_idx):
            storage = self.weights
            def capturing_forward(x, seq_id):
                qkv_BLD3 = attn_mod.layernorm_qkv(x)
                query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
                query_BLD, key_BLD = (
                    attn_mod.q_ln(query_BLD).to(query_BLD.dtype),
                    attn_mod.k_ln(key_BLD).to(query_BLD.dtype),
                )
                query_BLD, key_BLD = attn_mod._apply_rotary(query_BLD, key_BLD)
                reshaper = functools.partial(
                    einops.rearrange, pattern="b s (h d) -> b h s d", h=attn_mod.n_heads
                )
                query_BHLD, key_BHLD, value_BHLD = map(
                    reshaper, (query_BLD, key_BLD, value_BLD)
                )
                d_head = attn_mod.d_head
                attn_logits = torch.matmul(
                    query_BHLD, key_BHLD.transpose(-2, -1)
                ) / (d_head ** 0.5)
                if seq_id is not None:
                    mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
                    mask_BHLL = mask_BLL.unsqueeze(1)
                    attn_logits = attn_logits.masked_fill(~mask_BHLL, float("-inf"))
                else:
                    mask_BHLL = None
                attn_w = torch.softmax(attn_logits, dim=-1)
                storage[layer_idx] = attn_w.detach().float().cpu()
                if mask_BHLL is not None:
                    context_BHLD = F.scaled_dot_product_attention(
                        query_BHLD, key_BHLD, value_BHLD, mask_BHLL
                    )
                else:
                    context_BHLD = F.scaled_dot_product_attention(
                        query_BHLD, key_BHLD, value_BHLD
                    )
                context_BLD = einops.rearrange(context_BHLD, "b h s d -> b s (h d)")
                return attn_mod.out_proj(context_BLD)
            return capturing_forward

        def remove(self):
            for li, orig in self._orig_forwards.items():
                self.model.transformer.blocks[li].attn.forward = orig
            self._orig_forwards.clear()
            self.weights.clear()

    capture = AttentionCapture(model, target_layers)

    # Compute functional attention enrichment per head
    # enrichment = mean_attn_to_functional / mean_attn_to_non_functional
    results_s = {l: np.zeros(n_heads) for l in target_layers}
    results_sst = {l: np.zeros(n_heads) for l in target_layers}
    count = 0

    start_time = time.time()

    for idx, acc in enumerate(subset_accs):
        seq = seq_dict[acc]
        pdb_path = available_structures[acc]
        m = metadata.get(acc, {})
        feats = m.get("features", [])

        try:
            protein = ESMProtein.from_pdb(pdb_path)
            if protein.sequence is None:
                continue
            pdb_seq = protein.sequence
            seq_len = len(pdb_seq)

            # Build functional site mask (0-indexed)
            func_mask = np.zeros(seq_len, dtype=bool)
            for feat in feats:
                start = feat.get("start", 1) - 1
                end = feat.get("end", start + 1)
                func_mask[start:end] = True

            if func_mask.sum() == 0 or func_mask.sum() == seq_len:
                continue

            # S-only
            inputs_s = tokenize_sequence(pdb_seq, tokenizers, device=device)
            capture.weights.clear()
            capture.install()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**inputs_s)
            attn_s = {l: w.float().numpy() for l, w in capture.weights.items()}
            capture.remove()

            # S+St
            tensor = model.encode(protein)
            kwargs_sst = {"sequence_tokens": tensor.sequence.unsqueeze(0).to(device)}
            if tensor.structure is not None:
                kwargs_sst["structure_tokens"] = tensor.structure.unsqueeze(0).to(device)

            capture.weights.clear()
            capture.install()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(**kwargs_sst)
            attn_sst = {l: w.float().numpy() for l, w in capture.weights.items()}
            capture.remove()

            for li in target_layers:
                if li not in attn_s or li not in attn_sst:
                    continue
                # (1, n_heads, L+2, L+2) -> (n_heads, L, L) strip BOS/EOS
                ws = attn_s[li][0, :, 1:seq_len+1, 1:seq_len+1]
                wst = attn_sst[li][0, :, 1:seq_len+1, 1:seq_len+1]

                for h in range(n_heads):
                    # Mean attention TO functional vs non-functional positions
                    attn_to_func_s = ws[h, :, func_mask].mean()
                    attn_to_nonfunc_s = ws[h, :, ~func_mask].mean()
                    if attn_to_nonfunc_s > 1e-10:
                        results_s[li][h] += attn_to_func_s / attn_to_nonfunc_s

                    attn_to_func_sst = wst[h, :, func_mask].mean()
                    attn_to_nonfunc_sst = wst[h, :, ~func_mask].mean()
                    if attn_to_nonfunc_sst > 1e-10:
                        results_sst[li][h] += attn_to_func_sst / attn_to_nonfunc_sst

            count += 1
            if count % 50 == 0:
                elapsed = time.time() - start_time
                log.info(f"  Functional: {count}/{len(subset_accs)} ({count/elapsed:.1f} prot/s)")

        except Exception as e:
            if count < 5:
                log.warning(f"  {acc}: {e}")

    if count == 0:
        log.error("No proteins processed for functional attention")
        return

    # Average
    for li in target_layers:
        results_s[li] /= count
        results_sst[li] /= count

    del model
    torch.cuda.empty_cache()

    output = {
        "num_proteins": count,
        "layers": target_layers,
        "n_heads": n_heads,
        "enrichment_s": {str(l): results_s[l].tolist() for l in target_layers},
        "enrichment_sst": {str(l): results_sst[l].tolist() for l in target_layers},
    }
    with open(func_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"\nStep 8 complete ({count} proteins)")
    for li in target_layers:
        max_s = results_s[li].max()
        max_sst = results_sst[li].max()
        log.info(f"  Layer {li}: max enrichment S={max_s:.2f}, S+St={max_sst:.2f}")


# ============================================================
# Step 9: Train 6 SAEs
# ============================================================
def step9_train_saes():
    """Train TopK SAEs on scaled activations: S/S+St × layers 16/33/42."""
    log.info("=" * 60)
    log.info("STEP 9: Training SAEs (scaled)")
    log.info("=" * 60)

    import torch
    import h5py
    from sae.model import SAEConfig
    from sae.train import SAETrainer, TrainConfig

    device = "cuda:0"

    for condition in ["S", "S+St"]:
        for layer_idx in TARGET_LAYERS:
            safe_cond = condition.replace("+", "_")
            output_path = MODEL_DIR / f"{safe_cond}_layer_{layer_idx}_topk"

            # Check if already trained
            if (output_path / "final.pt").exists():
                log.info(f"SAE {condition} L{layer_idx}: already trained, skipping")
                continue

            h5_path = ACT_MM_DIR / f"{condition}_layer_{layer_idx}.h5"
            if not h5_path.exists():
                log.warning(f"Missing activations: {h5_path}")
                continue

            with h5py.File(h5_path, "r") as f:
                activations = torch.tensor(f["activations"][:], dtype=torch.float32)

            log.info(f"\n--- Training SAE: {condition} layer {layer_idx} ---")
            log.info(f"  Activations: {activations.shape}")

            n = len(activations)
            perm = torch.randperm(n)
            n_train = int(0.9 * n)
            train_acts = activations[perm[:n_train]]
            val_acts = activations[perm[n_train:]]

            hidden_dim = activations.shape[1]
            del activations

            sae_config = SAEConfig(
                input_dim=hidden_dim,
                expansion_factor=8,
                k=64,
                architecture="topk",
            )

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

            summary = trainer.train_on_tensor(train_acts, val_acts)
            trainer.save(output_path / "final.pt")

            log.info(f"  Done: recon={summary['final_recon']:.4f}, "
                     f"L0={summary['final_l0']:.1f}, dead={summary['dead_frac']:.3f}")

            with open(output_path / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)

            del train_acts, val_acts
            torch.cuda.empty_cache()

    log.info("\nStep 9 complete: all SAEs trained.")


# ============================================================
# Step 10: SS probing (S vs S+St at target layers + S-only 9 layers)
# ============================================================
def step10_ss_probing(dssp_results: dict):
    """Secondary structure probing on scaled data."""
    log.info("=" * 60)
    log.info("STEP 10: SS Probing (scaled)")
    log.info("=" * 60)

    import numpy as np
    import h5py
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    probe_dir = RESULTS_DIR / "ss_probing"
    probe_dir.mkdir(parents=True, exist_ok=True)

    probe_path = probe_dir / "ss_probing_results.json"
    if probe_path.exists():
        log.info("Step 10: SS probing results exist, skipping")
        return

    ss3_map = {"H": 0, "E": 1, "C": 2}
    results = []

    # --- S vs S+St at target layers ---
    for condition in ["S", "S+St"]:
        index_path = ACT_MM_DIR / f"{condition}_residue_index.json"
        if not index_path.exists():
            continue
        with open(index_path) as f:
            residue_index = json.load(f)

        # Build SS labels
        labels = []
        valid_mask = []
        for entry in residue_index:
            acc = entry["accession"]
            dssp = dssp_results.get(acc, [])
            for pos in range(entry["length"]):
                if pos < len(dssp):
                    ss3 = dssp[pos].get("ss3", "C")
                    labels.append(ss3_map.get(ss3, 2))
                    valid_mask.append(True)
                else:
                    labels.append(2)
                    valid_mask.append(False)

        labels = np.array(labels)
        valid_mask = np.array(valid_mask)
        log.info(f"  {condition}: {valid_mask.sum()} residues with DSSP labels")

        for layer_idx in TARGET_LAYERS:
            h5_path = ACT_MM_DIR / f"{condition}_layer_{layer_idx}.h5"
            if not h5_path.exists():
                continue

            with h5py.File(h5_path, "r") as f:
                X = f["activations"][:][valid_mask]
            y = labels[valid_mask]

            # Train/test split (80/20)
            n = len(X)
            n_train = int(0.8 * n)
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                     multi_class="multinomial", n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc_score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            results.append({
                "condition": condition,
                "layer": layer_idx,
                "accuracy": round(acc_score, 4),
                "f1_macro": round(f1, 4),
                "n_train": n_train,
                "n_test": n - n_train,
            })
            log.info(f"  {condition} L{layer_idx}: acc={acc_score:.3f}, f1={f1:.3f}")

    # --- S-only at 9 layers ---
    log.info("\n--- S-only SS probing at 9 layers ---")
    s_index_path = ACT_S_DIR / "residue_index.json"
    if s_index_path.exists():
        with open(s_index_path) as f:
            residue_index = json.load(f)

        labels = []
        valid_mask = []
        for entry in residue_index:
            acc = entry["accession"]
            dssp = dssp_results.get(acc, [])
            for pos in range(entry["length"]):
                if pos < len(dssp):
                    ss3 = dssp[pos].get("ss3", "C")
                    labels.append(ss3_map.get(ss3, 2))
                    valid_mask.append(True)
                else:
                    labels.append(2)
                    valid_mask.append(False)

        labels = np.array(labels)
        valid_mask = np.array(valid_mask)
        log.info(f"  S-only (9 layers): {valid_mask.sum()} residues with DSSP labels")

        for layer_idx in LAYERS_9:
            h5_path = ACT_S_DIR / f"layer_{layer_idx}.h5"
            if not h5_path.exists():
                continue

            with h5py.File(h5_path, "r") as f:
                X = f["activations"][:][valid_mask]
            y = labels[valid_mask]

            n = len(X)
            n_train = int(0.8 * n)
            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                     multi_class="multinomial", n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc_score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")

            results.append({
                "condition": "S_9layers",
                "layer": layer_idx,
                "accuracy": round(acc_score, 4),
                "f1_macro": round(f1, 4),
                "n_train": n_train,
                "n_test": n - n_train,
            })
            log.info(f"  S-only L{layer_idx}: acc={acc_score:.3f}, f1={f1:.3f}")

    with open(probe_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nStep 10 complete: {len(results)} probe results saved")


# ============================================================
# Step 11: SAE annotation + DSSP structural
# ============================================================
def step11_sae_annotation(dssp_results: dict, metadata: dict, seq_dict: dict):
    """Annotate SAE features with AA enrichment, functional site, and SS enrichment."""
    log.info("=" * 60)
    log.info("STEP 11: SAE Feature Annotation (scaled)")
    log.info("=" * 60)

    import torch
    import numpy as np
    import h5py
    sys.path.insert(0, str(ROOT / "src"))
    from sae.model import SAEConfig, build_sae

    annot_dir = RESULTS_DIR / "modality_saes"
    annot_dir.mkdir(parents=True, exist_ok=True)

    annot_path = annot_dir / "all_annotations.json"
    if annot_path.exists():
        log.info("Step 11: Annotations exist, skipping")
        return

    aa_set = list("ACDEFGHIKLMNPQRSTVWY")
    ss3_set = ["H", "E", "C"]
    all_annotations = {}

    for condition in ["S", "S+St"]:
        for layer_idx in TARGET_LAYERS:
            safe_cond = condition.replace("+", "_")
            ckpt_path = MODEL_DIR / f"{safe_cond}_layer_{layer_idx}_topk" / "final.pt"
            if not ckpt_path.exists():
                log.warning(f"Missing SAE checkpoint: {ckpt_path}")
                continue

            h5_path = ACT_MM_DIR / f"{condition}_layer_{layer_idx}.h5"
            index_path = ACT_MM_DIR / f"{condition}_residue_index.json"
            if not h5_path.exists() or not index_path.exists():
                continue

            log.info(f"\n--- Annotating: {condition} layer {layer_idx} ---")

            # Load SAE
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sae_config = ckpt["sae_config"]
            sae = build_sae(sae_config)
            sae.load_state_dict(ckpt["model_state_dict"])
            sae.eval()
            sae = sae.to("cuda:0")

            # Load activations
            with h5py.File(h5_path, "r") as f:
                activations = torch.tensor(f["activations"][:], dtype=torch.float32)
            with open(index_path) as f:
                residue_index = json.load(f)

            # Build per-residue labels
            aa_labels = []
            func_labels = []
            ss_labels = []
            for entry in residue_index:
                acc = entry["accession"]
                seq = seq_dict.get(acc, "")
                prot = metadata.get(acc, {})
                dssp = dssp_results.get(acc, [])

                for pos in range(entry["length"]):
                    aa_labels.append(seq[pos] if pos < len(seq) else "X")

                    is_func = False
                    for feat in prot.get("features", []):
                        start = feat.get("start", 0) - 1
                        end = feat.get("end", 0)
                        if start <= pos < end:
                            is_func = True
                            break
                    func_labels.append(is_func)

                    ss = dssp[pos]["ss3"] if pos < len(dssp) else "C"
                    ss_labels.append(ss)

            aa_labels = np.array(aa_labels)
            func_labels = np.array(func_labels)
            ss_labels = np.array(ss_labels)

            # Encode through SAE
            all_features = []
            batch_size = 8192
            for i in range(0, len(activations), batch_size):
                batch = activations[i:i + batch_size].to("cuda:0")
                with torch.no_grad():
                    z = sae.encode(batch)
                all_features.append(z.cpu())
            all_features = torch.cat(all_features, dim=0)

            feature_freq = (all_features > 0).float().mean(dim=0)
            active_count = (feature_freq > 0.001).sum().item()
            dead_count = (feature_freq == 0).sum().item()
            log.info(f"  Active: {active_count}, Dead: {dead_count}")

            # Annotate top 200 features
            top_indices = feature_freq.argsort(descending=True)[:200]
            annotations = []

            bg_aa = {aa: (aa_labels == aa).sum() / len(aa_labels) for aa in aa_set}
            bg_func = func_labels.mean()
            bg_ss = {ss: (ss_labels == ss).sum() / len(ss_labels) for ss in ss3_set}

            for feat_idx in top_indices:
                feat_idx = feat_idx.item()
                feat_acts = all_features[:, feat_idx]
                active_mask = feat_acts > 0

                if active_mask.sum() < 10:
                    continue

                active_np = active_mask.numpy()
                total_active = active_np.sum()

                # AA enrichment
                enrichments = {}
                for aa in aa_set:
                    frac = (aa_labels[active_np] == aa).sum() / total_active
                    if bg_aa[aa] > 0:
                        enrichments[aa] = round(frac / bg_aa[aa], 2)
                top_enriched = sorted(enrichments.items(), key=lambda x: -x[1])[:3]

                # Functional enrichment
                func_in_active = func_labels[active_np].mean()
                func_enrichment = func_in_active / bg_func if bg_func > 0 else 0

                # SS enrichment
                ss_enrichments = {}
                for ss in ss3_set:
                    frac = (ss_labels[active_np] == ss).sum() / total_active
                    if bg_ss[ss] > 0:
                        ss_enrichments[ss] = round(frac / bg_ss[ss], 2)

                annotations.append({
                    "feature_idx": feat_idx,
                    "activation_freq": round(feature_freq[feat_idx].item(), 5),
                    "top_enriched_aa": [(aa, e) for aa, e in top_enriched],
                    "functional_site_enrichment": round(func_enrichment, 2),
                    "ss_enrichment": ss_enrichments,
                })

            key = f"{safe_cond}_layer_{layer_idx}"
            all_annotations[key] = annotations

            strong_aa = sum(1 for a in annotations if any(e > 3.0 for _, e in a["top_enriched_aa"]))
            func_features = sum(1 for a in annotations if a["functional_site_enrichment"] > 2.0)
            log.info(f"  Annotated: {len(annotations)} | AA>3x: {strong_aa} | Func>2x: {func_features}")

            # Save individual
            with open(annot_dir / f"{key}_annotations.json", "w") as f:
                json.dump(annotations, f, indent=2)

            del sae, activations, all_features
            torch.cuda.empty_cache()

    with open(annot_path, "w") as f:
        json.dump(all_annotations, f, indent=2)

    log.info(f"\nStep 11 complete")


# ============================================================
# Step 12: Co-activation analysis
# ============================================================
def step12_coactivation():
    """Analyze feature co-activation patterns in scaled SAEs."""
    log.info("=" * 60)
    log.info("STEP 12: Co-activation Analysis (scaled)")
    log.info("=" * 60)

    import torch
    import numpy as np
    import h5py
    from sae.model import build_sae

    coact_dir = RESULTS_DIR / "coactivation"
    coact_dir.mkdir(parents=True, exist_ok=True)

    coact_path = coact_dir / "coactivation_results.json"
    if coact_path.exists():
        log.info("Step 12: Co-activation results exist, skipping")
        return

    results = {}

    for condition in ["S", "S+St"]:
        for layer_idx in TARGET_LAYERS:
            safe_cond = condition.replace("+", "_")
            ckpt_path = MODEL_DIR / f"{safe_cond}_layer_{layer_idx}_topk" / "final.pt"
            h5_path = ACT_MM_DIR / f"{condition}_layer_{layer_idx}.h5"

            if not ckpt_path.exists() or not h5_path.exists():
                continue

            log.info(f"\n--- Co-activation: {condition} L{layer_idx} ---")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sae = build_sae(ckpt["sae_config"])
            sae.load_state_dict(ckpt["model_state_dict"])
            sae.eval()
            sae = sae.to("cuda:0")

            with h5py.File(h5_path, "r") as f:
                activations = torch.tensor(f["activations"][:], dtype=torch.float32)

            # Encode
            all_z = []
            for i in range(0, len(activations), 8192):
                batch = activations[i:i + 8192].to("cuda:0")
                with torch.no_grad():
                    z = sae.encode(batch)
                all_z.append((z > 0).cpu().float())
            all_z = torch.cat(all_z, dim=0)

            # Feature frequencies
            freq = all_z.mean(dim=0)
            active_features = (freq > 0.001).nonzero().squeeze(-1)
            log.info(f"  Active features: {len(active_features)}")

            if len(active_features) < 20:
                continue

            # Top 200 features by frequency
            top_k = min(200, len(active_features))
            top_feat_indices = freq.argsort(descending=True)[:top_k]
            z_sub = all_z[:, top_feat_indices]

            # Co-activation matrix: Jaccard similarity
            n_sub = len(z_sub)
            co_mat = (z_sub.T @ z_sub) / n_sub
            freq_sub = z_sub.mean(dim=0)

            # Find top co-activation pairs
            pairs = []
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    joint = co_mat[i, j].item()
                    expected = freq_sub[i].item() * freq_sub[j].item()
                    if expected > 0:
                        pmi = np.log(joint / expected + 1e-10)
                        pairs.append({
                            "feat_i": top_feat_indices[i].item(),
                            "feat_j": top_feat_indices[j].item(),
                            "joint_freq": round(joint, 5),
                            "pmi": round(pmi, 3),
                        })

            pairs.sort(key=lambda x: -x["pmi"])
            top_pairs = pairs[:50]

            key = f"{safe_cond}_layer_{layer_idx}"
            results[key] = {
                "num_active": len(active_features),
                "top_pairs": top_pairs,
            }

            log.info(f"  Top co-activation PMI: {top_pairs[0]['pmi']:.3f}" if top_pairs else "  No pairs found")

            del sae, activations, all_z
            torch.cuda.empty_cache()

    with open(coact_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nStep 12 complete")


# ============================================================
# Step 13: Decoder geometry analysis
# ============================================================
def step13_decoder_geometry():
    """Analyze SAE decoder weight geometry."""
    log.info("=" * 60)
    log.info("STEP 13: Decoder Geometry (scaled)")
    log.info("=" * 60)

    import torch
    import numpy as np
    from sae.model import build_sae

    geo_dir = RESULTS_DIR / "decoder_geometry"
    geo_dir.mkdir(parents=True, exist_ok=True)

    geo_path = geo_dir / "decoder_geometry.json"
    if geo_path.exists():
        log.info("Step 13: Decoder geometry results exist, skipping")
        return

    results = {}

    for condition in ["S", "S+St"]:
        for layer_idx in TARGET_LAYERS:
            safe_cond = condition.replace("+", "_")
            ckpt_path = MODEL_DIR / f"{safe_cond}_layer_{layer_idx}_topk" / "final.pt"

            if not ckpt_path.exists():
                continue

            log.info(f"\n--- Decoder geometry: {condition} L{layer_idx} ---")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sae = build_sae(ckpt["sae_config"])
            sae.load_state_dict(ckpt["model_state_dict"])
            sae.eval()

            # Get decoder weights (dict_size, input_dim)
            W_dec = sae.decoder.weight.detach().numpy()
            if W_dec.shape[0] < W_dec.shape[1]:
                W_dec = W_dec.T  # ensure (dict_size, input_dim)

            # Normalize
            norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
            W_norm = W_dec / (norms + 1e-8)

            # Cosine similarity matrix (sample for efficiency)
            n_feat = W_norm.shape[0]
            if n_feat > 2000:
                idx = np.random.choice(n_feat, 2000, replace=False)
                W_sample = W_norm[idx]
            else:
                W_sample = W_norm

            cos_sim = W_sample @ W_sample.T
            np.fill_diagonal(cos_sim, 0)

            # Stats
            mean_cos = cos_sim.mean()
            max_cos = cos_sim.max()
            std_cos = cos_sim.std()

            # Effective dimensionality (via singular values)
            _, S, _ = np.linalg.svd(W_dec[:min(1000, n_feat)], full_matrices=False)
            S_norm = S / S.sum()
            eff_dim = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-10)))

            key = f"{safe_cond}_layer_{layer_idx}"
            results[key] = {
                "n_features": n_feat,
                "mean_cosine_sim": round(float(mean_cos), 5),
                "max_cosine_sim": round(float(max_cos), 5),
                "std_cosine_sim": round(float(std_cos), 5),
                "effective_dimensionality": round(float(eff_dim), 1),
                "decoder_norm_mean": round(float(norms.mean()), 5),
                "decoder_norm_std": round(float(norms.std()), 5),
            }

            log.info(f"  Mean cos sim: {mean_cos:.5f}, Eff dim: {eff_dim:.1f}")

            # Save decoder coordinates for visualization (PCA)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            coords = pca.fit_transform(W_norm[:min(5000, n_feat)])
            np.savez_compressed(
                geo_dir / f"{key}_coords.npz",
                coords=coords, explained_var=pca.explained_variance_ratio_,
            )

            del sae

    with open(geo_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nStep 13 complete")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale dataset pipeline")
    parser.add_argument("--step", type=int, default=0,
                        help="Run specific step (1-13), or 0 for all")
    parser.add_argument("--from-step", type=int, default=1,
                        help="Start from this step (when --step=0)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("SCALED DATASET PIPELINE (199 → ~5,000 proteins)")
    log.info("=" * 60)

    overall_start = time.time()

    try:
        # Step 1: Curate proteins
        if args.step in (0, 1) and args.from_step <= 1:
            seq_dict, metadata = step1_curate_proteins()
        else:
            with open(SEQ_DIR / "sequences.json") as f:
                seq_dict = json.load(f)
            with open(ANNOT_DIR / "metadata.json") as f:
                metadata = json.load(f)
            log.info(f"Loaded existing data: {len(seq_dict)} proteins")

        # Step 2: Download structures
        if args.step in (0, 2) and args.from_step <= 2:
            available = step2_download_structures(seq_dict)
        else:
            struct_map_path = DATA_DIR / "structure_map.json"
            if struct_map_path.exists():
                with open(struct_map_path) as f:
                    smap = json.load(f)
                available = {acc: str(STRUCT_DIR / f"{acc}.pdb")
                             for acc in smap["available"]
                             if (STRUCT_DIR / f"{acc}.pdb").exists()}
            else:
                available = {f.stem: str(f) for f in STRUCT_DIR.glob("*.pdb")}
            log.info(f"Loaded existing structures: {len(available)}")

        # Step 3: S-only activations
        if args.step in (0, 3) and args.from_step <= 3:
            step3_extract_s_only(seq_dict)

        # Step 4: S+St activations
        if args.step in (0, 4) and args.from_step <= 4:
            step4_extract_multimodal(seq_dict, available)

        # Step 5: DSSP
        if args.step in (0, 5) and args.from_step <= 5:
            dssp_results = step5_run_dssp(available)
        else:
            dssp_path = ANNOT_DIR / "dssp_annotations.json"
            if dssp_path.exists():
                with open(dssp_path) as f:
                    dssp_results = json.load(f)
            else:
                dssp_results = {}

        # Step 6: CKA
        if args.step in (0, 6) and args.from_step <= 6:
            step6_cka_modality(seq_dict, available, metadata)

        # Step 7: Attention atlas
        if args.step in (0, 7) and args.from_step <= 7:
            step7_attention_atlas(seq_dict, available)

        # Step 8: Functional attention
        if args.step in (0, 8) and args.from_step <= 8:
            step8_functional_attention(seq_dict, available, metadata)

        # Step 9: Train SAEs
        if args.step in (0, 9) and args.from_step <= 9:
            step9_train_saes()

        # Step 10: SS probing
        if args.step in (0, 10) and args.from_step <= 10:
            step10_ss_probing(dssp_results)

        # Step 11: SAE annotation
        if args.step in (0, 11) and args.from_step <= 11:
            step11_sae_annotation(dssp_results, metadata, seq_dict)

        # Step 12: Co-activation
        if args.step in (0, 12) and args.from_step <= 12:
            step12_coactivation()

        # Step 13: Decoder geometry
        if args.step in (0, 13) and args.from_step <= 13:
            step13_decoder_geometry()

        elapsed = time.time() - overall_start
        log.info(f"\n{'='*60}")
        log.info(f"PIPELINE COMPLETE in {elapsed/3600:.1f} hours")
        log.info(f"{'='*60}")

    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        raise
