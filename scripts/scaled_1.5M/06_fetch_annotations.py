#!/usr/bin/env python3
"""Fetch functional annotations from UniProt for the expanded eval set.

Fetches: GO terms, functional features (active sites, binding sites, etc.),
Pfam domains, EC numbers, keywords, InterPro, PDB IDs.

Uses UniProt REST API in batches of 100 accessions.

Usage:
    ./env/bin/python scripts/scaled_1.5M/06_fetch_annotations.py
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
os.chdir(ROOT)

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fetch_annot")

EVAL_DIR = ROOT / "data" / "eval_expanded"
BATCH_SIZE = 25  # UniProt API batch size (keep URL short)
UNIPROT_API = "https://rest.uniprot.org/uniprotkb"


def fetch_single_accession(accession):
    """Fetch full entry for a single accession from UniProt REST API."""
    resp = requests.get(f"{UNIPROT_API}/{accession}.json", timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_features(entry):
    """Extract functional site features from a UniProt entry."""
    features = []
    for feat in entry.get("features", []):
        feat_type = feat.get("type", "")
        if feat_type in [
            "Active site", "Binding site", "Site", "Metal binding",
            "Modified residue", "Lipidation", "Glycosylation site",
            "Disulfide bond", "Signal peptide", "Transit peptide",
            "Transmembrane",
        ]:
            location = feat.get("location", {})
            start = location.get("start", {}).get("value")
            end = location.get("end", {}).get("value")
            if start is not None and end is not None:
                features.append({
                    "type": feat_type,
                    "start": int(start),
                    "end": int(end),
                    "description": feat.get("description", ""),
                })
    return features


def parse_go_terms(entry):
    """Extract GO terms from UniProt entry."""
    go_terms = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == "GO":
            go_id = xref.get("id", "")
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            term = props.get("GoTerm", "")
            evidence = props.get("GoEvidenceType", "")
            go_terms.append({
                "id": go_id,
                "term": term,
                "evidence": evidence,
            })
    return go_terms


def parse_xrefs(entry, db_name):
    """Extract cross-references for a specific database."""
    refs = []
    for xref in entry.get("uniProtKBCrossReferences", []):
        if xref.get("database") == db_name:
            ref_id = xref.get("id", "")
            props = {p["key"]: p["value"] for p in xref.get("properties", [])}
            refs.append({"id": ref_id, **props})
    return refs


def main():
    # Load eval set metadata
    meta_path = EVAL_DIR / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    accessions = list(metadata.keys())
    log.info(f"Fetching annotations for {len(accessions)} proteins...")

    # Process individually with concurrent fetching
    from concurrent.futures import ThreadPoolExecutor, as_completed

    failed = []
    total_fetched = 0

    # Skip already-annotated proteins (for resume)
    to_fetch = [acc for acc in accessions if not metadata[acc].get("go_terms")]
    log.info(f"Need to fetch: {len(to_fetch)} (skipping {len(accessions) - len(to_fetch)} already done)")

    def process_one(acc):
        """Fetch and parse one accession."""
        entry = fetch_single_accession(acc)

        protein_name = ""
        if entry.get("proteinDescription", {}).get("recommendedName"):
            protein_name = entry["proteinDescription"]["recommendedName"].get(
                "fullName", {}).get("value", "")
        elif entry.get("proteinDescription", {}).get("submissionNames"):
            protein_name = entry["proteinDescription"]["submissionNames"][0].get(
                "fullName", {}).get("value", "")

        organism = entry.get("organism", {}).get("scientificName", "")
        taxid = str(entry.get("organism", {}).get("taxonId", ""))

        go_terms = parse_go_terms(entry)
        features = parse_features(entry)
        pfam = parse_xrefs(entry, "Pfam")
        interpro = parse_xrefs(entry, "InterPro")
        pdb_ids = [x["id"] for x in parse_xrefs(entry, "PDB")]

        ec_numbers = []
        if entry.get("proteinDescription", {}).get("recommendedName", {}).get("ecNumbers"):
            ec_numbers = [e["value"] for e in
                          entry["proteinDescription"]["recommendedName"]["ecNumbers"]]

        keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]

        return {
            "acc": acc,
            "protein_name": protein_name,
            "organism": organism,
            "taxid": taxid,
            "go_terms": go_terms,
            "features": features,
            "pfam": [p["id"] for p in pfam],
            "interpro": [p["id"] for p in interpro],
            "pdb_ids": pdb_ids,
            "ec_numbers": ec_numbers,
            "keywords": keywords,
            "n_functional_sites": len([f for f in features
                                        if f["type"] in ["Active site", "Binding site",
                                                           "Metal binding", "Site"]]),
        }

    # Use 8 threads for concurrent fetching (respect rate limits)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_one, acc): acc for acc in to_fetch}

        for future in as_completed(futures):
            acc = futures[future]
            try:
                result = future.result()
                metadata[acc].update({k: v for k, v in result.items() if k != "acc"})
                total_fetched += 1
            except Exception as e:
                failed.append(acc)

            if (total_fetched + len(failed)) % 500 == 0:
                log.info(f"  Progress: {total_fetched} annotated, {len(failed)} failed "
                         f"/ {len(to_fetch)} total")

            # Save checkpoint every 2000
            if total_fetched % 2000 == 0 and total_fetched > 0:
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                log.info(f"  Checkpoint saved ({total_fetched} annotated)")

    # Retry failed ones with backoff
    if failed:
        log.info(f"\nRetrying {len(failed)} failed accessions...")
        retry_ok = 0
        for acc in failed:
            try:
                result = process_one(acc)
                metadata[acc].update({k: v for k, v in result.items() if k != "acc"})
                total_fetched += 1
                retry_ok += 1
                time.sleep(0.5)
            except Exception:
                pass
        log.info(f"  Retried: {retry_ok} succeeded, {len(failed) - retry_ok} still failed")

    # Save updated metadata
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary stats
    has_go = sum(1 for m in metadata.values() if m.get("go_terms"))
    has_features = sum(1 for m in metadata.values() if m.get("features"))
    has_func_sites = sum(1 for m in metadata.values() if m.get("n_functional_sites", 0) > 0)
    has_pfam = sum(1 for m in metadata.values() if m.get("pfam"))
    has_pdb = sum(1 for m in metadata.values() if m.get("pdb_ids"))
    has_ec = sum(1 for m in metadata.values() if m.get("ec_numbers"))

    log.info(f"\n{'='*60}")
    log.info("ANNOTATION SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"Total proteins: {len(metadata)}")
    log.info(f"Successfully annotated: {total_fetched}")
    log.info(f"Failed: {len(metadata) - total_fetched}")
    log.info(f"\nCoverage:")
    log.info(f"  GO terms:         {has_go:>6d} ({100*has_go/len(metadata):.1f}%)")
    log.info(f"  Any features:     {has_features:>6d} ({100*has_features/len(metadata):.1f}%)")
    log.info(f"  Functional sites: {has_func_sites:>6d} ({100*has_func_sites/len(metadata):.1f}%)")
    log.info(f"  Pfam:             {has_pfam:>6d} ({100*has_pfam/len(metadata):.1f}%)")
    log.info(f"  PDB structures:   {has_pdb:>6d} ({100*has_pdb/len(metadata):.1f}%)")
    log.info(f"  EC numbers:       {has_ec:>6d} ({100*has_ec/len(metadata):.1f}%)")

    # Feature type distribution
    feat_types = Counter()
    for m in metadata.values():
        for f in m.get("features", []):
            feat_types[f["type"]] += 1
    log.info(f"\nFeature types:")
    for ft, n in feat_types.most_common():
        log.info(f"  {ft:25s}: {n:>6d}")

    log.info(f"\nSaved to {meta_path}")


if __name__ == "__main__":
    main()
