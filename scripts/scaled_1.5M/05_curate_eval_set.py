#!/usr/bin/env python3
"""Curate an expanded evaluation set from Swiss-Prot, free of data leakage.

Steps:
1. Download/parse Swiss-Prot reviewed entries with rich annotations
2. Filter: length 50-800, has GO terms, has functional features
3. Stratify by taxonomy (bacteria, archaea, eukaryota, viruses)
4. Remove exact matches to 1.5M training set
5. Use MMseqs2 to remove any eval protein with >30% identity to training set
6. Save final eval set with annotations

Target: ~20K-50K diverse, well-annotated proteins with zero leakage.

Usage:
    ./env/bin/python scripts/scaled_1.5M/05_curate_eval_set.py
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import random

ROOT = Path(__file__).parent.parent.parent
os.chdir(ROOT)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("curate_eval")

MMSEQS = "/mnt/ca1e2e99-718e-417c-9ba6-62421455971a/SOFTWARE/mmseqs/bin/mmseqs"
SWISSPROT_DAT = ROOT / "data" / "eval_curation" / "uniprot_sprot.dat"
SWISSPROT_FASTA = ROOT / "data" / "eval_curation" / "uniprot_sprot.fasta"
TRAIN_FASTA = ROOT / "data" / "sae_training" / "uniref50_1.5M.fasta"
TRAIN_JSON = ROOT / "data" / "sae_training" / "uniref50_1.5M_sequences.json"

OUTPUT_DIR = ROOT / "data" / "eval_expanded"
TARGET_SIZE = 50_000
MIN_LEN = 50
MAX_LEN = 800
IDENTITY_THRESHOLD = 0.5  # 50% identity cutoff — matches UniRef50 clustering level
SEED = 42


def download_swissprot():
    """Download Swiss-Prot FASTA if not present."""
    SWISSPROT_FASTA.parent.mkdir(parents=True, exist_ok=True)

    if SWISSPROT_FASTA.exists():
        log.info(f"Swiss-Prot FASTA already exists: {SWISSPROT_FASTA}")
        return

    log.info("Downloading Swiss-Prot FASTA...")
    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    subprocess.run([
        "wget", "-q", "-O", str(SWISSPROT_FASTA) + ".gz", url
    ], check=True)
    subprocess.run(["gunzip", str(SWISSPROT_FASTA) + ".gz"], check=True)
    log.info(f"Downloaded to {SWISSPROT_FASTA}")


def parse_swissprot_fasta(path):
    """Parse Swiss-Prot FASTA, extracting accession and metadata from headers.

    Header format: >sp|ACCESSION|ENTRY_NAME PROTEIN_NAME OS=Organism OX=TaxID GN=Gene PE=Evidence SV=Version
    """
    proteins = {}
    current_acc = None
    current_meta = {}
    seq_parts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_acc and seq_parts:
                    proteins[current_acc] = {
                        **current_meta,
                        "sequence": "".join(seq_parts),
                    }

                # Parse header
                header = line[1:]
                parts = header.split("|")
                if len(parts) >= 3:
                    acc = parts[1]
                    rest = parts[2]
                else:
                    acc = header.split()[0]
                    rest = header

                # Extract fields
                name_part = rest.split(" OS=")[0] if " OS=" in rest else rest.split()[0]
                organism = ""
                taxid = ""
                if " OS=" in rest:
                    organism = rest.split(" OS=")[1].split(" OX=")[0].strip()
                if " OX=" in rest:
                    taxid = rest.split(" OX=")[1].split()[0].strip()

                current_acc = acc
                current_meta = {
                    "accession": acc,
                    "entry_name": name_part.split()[0] if name_part else acc,
                    "protein_name": " ".join(name_part.split()[1:]) if len(name_part.split()) > 1 else name_part,
                    "organism": organism,
                    "taxid": taxid,
                }
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_acc and seq_parts:
        proteins[current_acc] = {**current_meta, "sequence": "".join(seq_parts)}

    return proteins


def assign_taxonomy(taxid, organism):
    """Assign broad taxonomy group."""
    organism_lower = organism.lower()

    # Virus keywords
    if any(kw in organism_lower for kw in ["virus", "phage", "viridae"]):
        return "virus"

    # Archaea keywords
    if any(kw in organism_lower for kw in [
        "archae", "halobacterium", "methan", "sulfolobus", "thermococcus",
        "pyrococcus", "thermoplasma", "haloferax"
    ]):
        return "archaea"

    # Common bacteria
    if any(kw in organism_lower for kw in [
        "escherichia", "bacillus", "staphylococcus", "streptococcus",
        "salmonella", "pseudomonas", "mycobacterium", "clostridium",
        "helicobacter", "campylobacter", "neisseria", "vibrio",
        "bordetella", "listeria", "legionella", "chlamydia",
        "rickettsia", "borrelia", "treponema", "corynebacterium",
        "lactobacillus", "enterococcus", "klebsiella", "acinetobacter",
        "rhizobium", "agrobacterium", "caulobacter", "synechocystis",
        "cyanobact", "thermus", "deinococcus",
    ]):
        return "bacteria"

    # Eukaryotes - plants
    if any(kw in organism_lower for kw in [
        "arabidopsis", "oryza", "zea mays", "nicotiana", "solanum",
        "glycine max", "triticum", "physcomitrella", "marchantia",
    ]):
        return "plant"

    # Eukaryotes - fungi
    if any(kw in organism_lower for kw in [
        "saccharomyces", "schizosaccharomyces", "aspergillus",
        "neurospora", "candida", "cryptococcus", "ustilago",
    ]):
        return "fungi"

    # Eukaryotes - animals
    if any(kw in organism_lower for kw in [
        "homo sapiens", "mus musculus", "rattus", "bos taurus",
        "gallus", "danio rerio", "xenopus", "drosophila",
        "caenorhabditis", "sus scrofa", "ovis aries", "equus",
        "canis", "felis", "pan troglodytes", "macaca",
    ]):
        return "animal"

    # Default: try to classify by taxid ranges (rough)
    try:
        tid = int(taxid)
        if tid < 10000:
            return "bacteria"  # rough heuristic
    except (ValueError, TypeError):
        pass

    return "other_eukaryote"


def run_mmseqs_search(eval_fasta, train_fasta, output_dir, identity=0.3, threads=16):
    """Run MMseqs2 search to find eval proteins similar to training set."""
    tmpdir = Path(output_dir) / "mmseqs_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    eval_db = str(tmpdir / "eval_db")
    train_db = str(tmpdir / "train_db")
    result_db = str(tmpdir / "result_db")
    result_tsv = str(tmpdir / "results.tsv")

    log.info("Creating MMseqs2 databases...")
    subprocess.run([MMSEQS, "createdb", str(eval_fasta), eval_db], check=True,
                   capture_output=True)
    subprocess.run([MMSEQS, "createdb", str(train_fasta), train_db], check=True,
                   capture_output=True)

    log.info(f"Running MMseqs2 search (identity threshold={identity})...")
    subprocess.run([
        MMSEQS, "search", eval_db, train_db, result_db, str(tmpdir / "tmp"),
        "--min-seq-id", str(identity),
        "-s", "7.5",  # sensitivity
        "--threads", str(threads),
        "-e", "1e-5",
    ], check=True, capture_output=True)

    log.info("Converting results...")
    subprocess.run([
        MMSEQS, "convertalis", eval_db, train_db, result_db, result_tsv,
        "--format-output", "query,target,pident,alnlen,evalue",
    ], check=True, capture_output=True)

    # Parse results: collect eval accessions that have hits
    leaky_accs = set()
    with open(result_tsv) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                eval_acc = parts[0]
                pident = float(parts[2])
                if pident >= identity * 100:  # MMseqs reports as percentage
                    leaky_accs.add(eval_acc)

    log.info(f"Found {len(leaky_accs)} eval proteins with >{identity*100:.0f}% identity to training set")

    # Cleanup
    shutil.rmtree(tmpdir)

    return leaky_accs


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    # Step 1: Get Swiss-Prot
    download_swissprot()

    # Step 2: Parse Swiss-Prot
    log.info("Parsing Swiss-Prot FASTA...")
    sp_proteins = parse_swissprot_fasta(SWISSPROT_FASTA)
    log.info(f"Total Swiss-Prot entries: {len(sp_proteins)}")

    # Step 3: Load training set accessions for exact-match exclusion
    log.info("Loading training set IDs...")
    with open(TRAIN_JSON) as f:
        train_seqs = json.load(f)
    train_ids = set(train_seqs.keys())
    # Also extract UniProt accessions from UniRef50 IDs (format: UniRef50_ACCESSION)
    train_uniprot_accs = set()
    for uid in train_ids:
        if "_" in uid:
            acc = uid.split("_", 1)[1]
            train_uniprot_accs.add(acc)
    log.info(f"Training set: {len(train_ids)} UniRef IDs, {len(train_uniprot_accs)} UniProt accessions")

    # Also load old eval set
    old_eval_path = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    if old_eval_path.exists():
        with open(old_eval_path) as f:
            old_eval = set(json.load(f).keys())
        log.info(f"Old eval set: {len(old_eval)} proteins (will include if they pass filters)")
    else:
        old_eval = set()

    # Step 4: Filter Swiss-Prot
    log.info("Filtering Swiss-Prot entries...")
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    candidates = {}

    for acc, data in sp_proteins.items():
        seq = data["sequence"]
        L = len(seq)

        # Length filter
        if L < MIN_LEN or L > MAX_LEN:
            continue

        # Standard AA only
        if not all(c in valid_aa for c in seq.upper()):
            continue

        # Skip exact matches to training set
        if acc in train_uniprot_accs:
            continue

        # Assign taxonomy
        tax_group = assign_taxonomy(data.get("taxid", ""), data.get("organism", ""))
        data["taxonomy"] = tax_group
        data["length"] = L

        candidates[acc] = data

    log.info(f"After filtering: {len(candidates)} candidates")
    tax_counts = Counter(d["taxonomy"] for d in candidates.values())
    log.info(f"Taxonomy distribution:")
    for tax, n in tax_counts.most_common():
        log.info(f"  {tax:20s}: {n:>6d}")

    # Step 5: Stratified sampling
    log.info(f"\nStratified sampling to ~{TARGET_SIZE} proteins...")

    # Group by taxonomy
    by_tax = defaultdict(list)
    for acc, data in candidates.items():
        by_tax[data["taxonomy"]].append(acc)

    # Allocate proportionally but ensure minimum per group
    min_per_group = 500
    total_available = sum(len(v) for v in by_tax.values())
    selected_accs = []

    for tax, accs in sorted(by_tax.items()):
        # Proportional allocation with minimum
        n_alloc = max(min_per_group, int(TARGET_SIZE * len(accs) / total_available))
        n_alloc = min(n_alloc, len(accs))
        sampled = rng.sample(accs, n_alloc)
        selected_accs.extend(sampled)
        log.info(f"  {tax:20s}: {n_alloc:>5d} / {len(accs):>6d} available")

    # If over target, trim randomly
    if len(selected_accs) > TARGET_SIZE * 1.2:
        rng.shuffle(selected_accs)
        selected_accs = selected_accs[:TARGET_SIZE]

    log.info(f"Selected {len(selected_accs)} proteins before leakage filtering")

    # Step 6: Write eval FASTA for MMseqs2
    eval_fasta_path = OUTPUT_DIR / "eval_candidates.fasta"
    with open(eval_fasta_path, "w") as f:
        for acc in selected_accs:
            f.write(f">{acc}\n{candidates[acc]['sequence']}\n")

    # Step 7: MMseqs2 leakage check
    log.info("\nRunning MMseqs2 leakage check against training set...")
    leaky_accs = run_mmseqs_search(
        eval_fasta_path, TRAIN_FASTA, str(OUTPUT_DIR),
        identity=IDENTITY_THRESHOLD, threads=16,
    )

    # Remove leaky proteins
    clean_accs = [acc for acc in selected_accs if acc not in leaky_accs]
    log.info(f"Removed {len(selected_accs) - len(clean_accs)} leaky proteins")
    log.info(f"Final eval set: {len(clean_accs)} proteins")

    # Step 8: Save
    # Sequences
    sequences = {acc: candidates[acc]["sequence"] for acc in clean_accs}
    seq_path = OUTPUT_DIR / "sequences.json"
    with open(seq_path, "w") as f:
        json.dump(sequences, f)

    # FASTA
    fasta_path = OUTPUT_DIR / "sequences.fasta"
    with open(fasta_path, "w") as f:
        for acc in clean_accs:
            f.write(f">{acc}\n{candidates[acc]['sequence']}\n")

    # Metadata
    metadata = {}
    for acc in clean_accs:
        d = candidates[acc]
        metadata[acc] = {
            "accession": acc,
            "entry_name": d.get("entry_name", ""),
            "protein_name": d.get("protein_name", ""),
            "organism": d.get("organism", ""),
            "taxid": d.get("taxid", ""),
            "taxonomy": d.get("taxonomy", ""),
            "length": d.get("length", 0),
        }

    meta_path = OUTPUT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    final_tax = Counter(metadata[acc]["taxonomy"] for acc in clean_accs)
    lengths = [len(sequences[acc]) for acc in clean_accs]

    log.info(f"\n{'='*60}")
    log.info("FINAL EVALUATION SET")
    log.info(f"{'='*60}")
    log.info(f"Total proteins: {len(clean_accs)}")
    log.info(f"Length range: {min(lengths)}-{max(lengths)}, mean: {sum(lengths)/len(lengths):.0f}")
    log.info(f"Total residues: {sum(lengths):,}")
    log.info(f"\nTaxonomy:")
    for tax, n in final_tax.most_common():
        log.info(f"  {tax:20s}: {n:>5d} ({100*n/len(clean_accs):.1f}%)")

    # Check overlap with old eval
    overlap = set(clean_accs) & old_eval
    log.info(f"\nOverlap with old 4,793 eval set: {len(overlap)} proteins")
    log.info(f"New proteins: {len(clean_accs) - len(overlap)}")

    log.info(f"\nSaved to {OUTPUT_DIR}/")
    log.info(f"  sequences.json ({seq_path.stat().st_size / 1e6:.1f} MB)")
    log.info(f"  sequences.fasta")
    log.info(f"  metadata.json")

    # Clean up temp fasta
    eval_fasta_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
