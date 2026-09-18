#!/usr/bin/env python3
"""Sample 1.5M proteins from UniRef50 for SAE training.

Filters: length 50-1022, no ambiguous residues, random sample.
Also excludes our 4,800 evaluation proteins to prevent data leakage.

Usage:
    ./env/bin/python scripts/scaled_1.5M/01_sample_uniref50.py
"""

import os
import sys
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
os.chdir(ROOT)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sample")

UNIREF50_FASTA = "/mnt/85740f55-8e9a-4214-9500-be446866627e/uniref50/uniref50.fasta"
TARGET_N = 1_500_000
MIN_LEN = 50
MAX_LEN = 1022
SEED = 42
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

OUTPUT_DIR = ROOT / "data" / "sae_training"
OUTPUT_FASTA = OUTPUT_DIR / "uniref50_1.5M.fasta"
OUTPUT_JSON = OUTPUT_DIR / "uniref50_1.5M_sequences.json"


def parse_fasta_streaming(path):
    """Yield (header, sequence) tuples from FASTA file."""
    header = None
    seq_parts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].split()[0]  # Take first word as ID
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load eval protein accessions to exclude
    eval_seqs_path = ROOT / "data" / "scaled" / "sequences" / "sequences.json"
    if eval_seqs_path.exists():
        with open(eval_seqs_path) as f:
            eval_accs = set(json.load(f).keys())
        log.info(f"Loaded {len(eval_accs)} evaluation protein accessions to exclude")
    else:
        eval_accs = set()
        log.warning("No evaluation sequences found, proceeding without exclusion")

    # First pass: count eligible proteins and collect indices
    log.info(f"Scanning {UNIREF50_FASTA}...")
    log.info(f"Filters: length {MIN_LEN}-{MAX_LEN}, standard AA only")

    eligible_indices = []
    total = 0
    skipped_len = 0
    skipped_aa = 0
    skipped_eval = 0

    for header, seq in parse_fasta_streaming(UNIREF50_FASTA):
        total += 1

        if total % 5_000_000 == 0:
            log.info(f"  Scanned {total/1e6:.1f}M sequences, {len(eligible_indices)} eligible...")

        # Extract UniRef ID and possible UniProt accession
        # UniRef50 headers: UniRef50_P12345
        uniref_id = header
        uniprot_acc = header.split("_", 1)[1] if "_" in header else header

        if uniprot_acc in eval_accs:
            skipped_eval += 1
            continue

        L = len(seq)
        if L < MIN_LEN or L > MAX_LEN:
            skipped_len += 1
            continue

        if not all(c in VALID_AA for c in seq.upper()):
            skipped_aa += 1
            continue

        eligible_indices.append(total - 1)  # 0-indexed position

    log.info(f"Scan complete: {total} total, {len(eligible_indices)} eligible")
    log.info(f"  Skipped: {skipped_len} (length), {skipped_aa} (non-standard AA), "
             f"{skipped_eval} (eval set)")

    if len(eligible_indices) < TARGET_N:
        log.warning(f"Only {len(eligible_indices)} eligible proteins, fewer than target {TARGET_N}")
        sample_indices = set(eligible_indices)
    else:
        rng = random.Random(SEED)
        sample_indices = set(rng.sample(eligible_indices, TARGET_N))

    log.info(f"Selected {len(sample_indices)} proteins")

    # Second pass: extract selected sequences
    log.info("Extracting selected sequences...")
    sequences = {}
    idx = 0
    n_written = 0

    with open(OUTPUT_FASTA, "w") as fasta_out:
        for header, seq in parse_fasta_streaming(UNIREF50_FASTA):
            if idx in sample_indices:
                seq = seq.upper()
                uniref_id = header
                fasta_out.write(f">{uniref_id}\n{seq}\n")
                sequences[uniref_id] = seq
                n_written += 1

                if n_written % 500_000 == 0:
                    log.info(f"  Written {n_written}/{len(sample_indices)}")

            idx += 1
            if n_written >= len(sample_indices):
                break

    # Save as JSON for easy loading
    with open(OUTPUT_JSON, "w") as f:
        json.dump(sequences, f)

    lengths = [len(s) for s in sequences.values()]
    log.info(f"\nDone! Saved {len(sequences)} proteins")
    log.info(f"  FASTA: {OUTPUT_FASTA}")
    log.info(f"  JSON: {OUTPUT_JSON}")
    log.info(f"  Length range: {min(lengths)}-{max(lengths)}, mean: {sum(lengths)/len(lengths):.0f}")
    log.info(f"  Total residues: {sum(lengths):,}")


if __name__ == "__main__":
    main()
