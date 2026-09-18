"""Download and curate pilot protein dataset from UniProt/Swiss-Prot.

Downloads a diverse set of well-annotated proteins for pipeline validation.
"""

import json
import gzip
import requests
import time
from pathlib import Path
from typing import Optional


UNIPROT_API = "https://rest.uniprot.org"


def search_uniprot(
    query: str,
    fields: list[str],
    size: int = 500,
    format: str = "json",
) -> list[dict]:
    """Search UniProt and return results.

    Args:
        query: UniProt query string
        fields: list of field names to retrieve
        size: max results
        format: response format

    Returns:
        List of result dicts
    """
    url = f"{UNIPROT_API}/uniprotkb/search"
    params = {
        "query": query,
        "fields": ",".join(fields),
        "size": min(size, 500),
        "format": format,
    }

    all_results = []
    while len(all_results) < size:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)

        # Follow pagination link
        link = response.headers.get("Link", "")
        if 'rel="next"' in link:
            url = link.split(";")[0].strip("<>")
            params = {}  # URL already contains params
        else:
            break

        time.sleep(0.5)  # Rate limiting

    return all_results[:size]


def download_pdb_structure(pdb_id: str, output_dir: Path, format: str = "cif") -> Optional[Path]:
    """Download a structure from RCSB PDB.

    Args:
        pdb_id: 4-letter PDB ID
        output_dir: directory to save file
        format: "cif" or "pdb"

    Returns:
        Path to downloaded file, or None on failure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "cif" if format == "cif" else "pdb"
    output_path = output_dir / f"{pdb_id.lower()}.{ext}"

    if output_path.exists():
        return output_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.{ext}.gz"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = gzip.decompress(response.content)
        output_path.write_bytes(content)
        return output_path
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")
        return None


def download_alphafold_structure(uniprot_id: str, output_dir: Path) -> Optional[Path]:
    """Download predicted structure from AlphaFold Database.

    Args:
        uniprot_id: UniProt accession
        output_dir: directory to save file

    Returns:
        Path to downloaded file, or None on failure
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"AF-{uniprot_id}-F1-model_v4.cif"

    if output_path.exists():
        return output_path

    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        output_path.write_text(response.text)
        return output_path
    except Exception as e:
        # Try PDB format as fallback
        url_pdb = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        try:
            response = requests.get(url_pdb, timeout=30)
            response.raise_for_status()
            output_path_pdb = output_dir / f"AF-{uniprot_id}-F1-model_v4.pdb"
            output_path_pdb.write_text(response.text)
            return output_path_pdb
        except Exception:
            print(f"Failed to download AlphaFold structure for {uniprot_id}: {e}")
            return None


def build_pilot_queries() -> list[dict]:
    """Build UniProt queries for a diverse pilot set.

    Returns queries designed to sample across:
    - Structural classes (all-alpha, all-beta, mixed)
    - Functional categories (enzymes, binding, transport, etc.)
    - Length ranges
    """
    queries = [
        {
            "name": "enzymes_short",
            "query": "(reviewed:true) AND (ec:*) AND (length:[50 TO 200]) AND (structure_3d:true)",
            "target_count": 100,
        },
        {
            "name": "enzymes_medium",
            "query": "(reviewed:true) AND (ec:*) AND (length:[200 TO 400]) AND (structure_3d:true)",
            "target_count": 100,
        },
        {
            "name": "enzymes_long",
            "query": "(reviewed:true) AND (ec:*) AND (length:[400 TO 800]) AND (structure_3d:true)",
            "target_count": 75,
        },
        {
            "name": "binding_proteins",
            "query": "(reviewed:true) AND (keyword:KW-0067) AND (length:[50 TO 500]) AND (structure_3d:true)",
            "target_count": 100,
        },
        {
            "name": "transporters",
            "query": "(reviewed:true) AND (keyword:KW-0813) AND (length:[100 TO 600]) AND (structure_3d:true)",
            "target_count": 75,
        },
        {
            "name": "dna_binding",
            "query": "(reviewed:true) AND (keyword:KW-0238) AND (length:[50 TO 400]) AND (structure_3d:true)",
            "target_count": 75,
        },
        {
            "name": "structural_proteins",
            "query": "(reviewed:true) AND (keyword:KW-0732) AND (length:[50 TO 500]) AND (structure_3d:true)",
            "target_count": 50,
        },
        {
            "name": "receptors",
            "query": "(reviewed:true) AND (keyword:KW-0675) AND (length:[100 TO 600]) AND (structure_3d:true)",
            "target_count": 75,
        },
        {
            "name": "small_proteins",
            "query": "(reviewed:true) AND (length:[30 TO 100]) AND (structure_3d:true)",
            "target_count": 100,
        },
        {
            "name": "diverse_medium",
            "query": "(reviewed:true) AND (length:[100 TO 500]) AND (structure_3d:true) AND (annotation_score:5)",
            "target_count": 250,
        },
    ]
    return queries


def curate_pilot_dataset(
    output_dir: str = "data/pilot",
    target_total: int = 1000,
) -> dict:
    """Download and curate the pilot protein dataset.

    Args:
        output_dir: output directory
        target_total: target number of proteins

    Returns:
        Summary dict with counts and paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "sequences").mkdir(exist_ok=True)
    (output_path / "structures").mkdir(exist_ok=True)
    (output_path / "annotations").mkdir(exist_ok=True)

    fields = [
        "accession",
        "id",
        "protein_name",
        "gene_names",
        "organism_name",
        "length",
        "sequence",
        "go_id",
        "go_p",
        "go_f",
        "ec",
        "keyword",
        "ft_act_site",
        "ft_binding",
        "ft_site",
        "xref_pdb",
        "xref_interpro",
        "xref_pfam",
    ]

    queries = build_pilot_queries()
    all_proteins = {}
    seen_accessions = set()

    for q in queries:
        print(f"Querying: {q['name']} (target: {q['target_count']})...")
        results = search_uniprot(q["query"], fields, size=q["target_count"] * 2)

        count = 0
        for protein in results:
            acc = protein.get("primaryAccession", "")
            if acc in seen_accessions:
                continue
            seen_accessions.add(acc)

            seq = protein.get("sequence", {}).get("value", "")
            if not seq or len(seq) < 30:
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

        print(f"  Got {count} proteins (total so far: {len(all_proteins)})")
        time.sleep(1)

        if len(all_proteins) >= target_total:
            break

    # Save sequences as FASTA
    fasta_path = output_path / "sequences" / "pilot_sequences.fasta"
    with open(fasta_path, "w") as f:
        for acc, prot in all_proteins.items():
            f.write(f">{acc} {prot['entry_name']} {prot['protein_name']}\n")
            seq = prot["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")

    # Save metadata as JSON
    metadata_path = output_path / "annotations" / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(all_proteins, f, indent=2)

    # Save a simple sequence-only dict for quick loading
    seq_dict = {acc: prot["sequence"] for acc, prot in all_proteins.items()}
    seq_path = output_path / "sequences" / "sequences.json"
    with open(seq_path, "w") as f:
        json.dump(seq_dict, f)

    summary = {
        "total_proteins": len(all_proteins),
        "categories": {q["name"]: sum(1 for p in all_proteins.values() if p["category"] == q["name"]) for q in queries},
        "length_stats": {
            "min": min(p["length"] for p in all_proteins.values()),
            "max": max(p["length"] for p in all_proteins.values()),
            "mean": sum(p["length"] for p in all_proteins.values()) / len(all_proteins),
        },
        "with_pdb": sum(1 for p in all_proteins.values() if p["pdb_ids"]),
        "with_go": sum(1 for p in all_proteins.values() if p["go_terms"]),
        "with_ec": sum(1 for p in all_proteins.values() if p["ec_numbers"]),
        "fasta_path": str(fasta_path),
        "metadata_path": str(metadata_path),
    }

    summary_path = output_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPilot dataset summary:")
    print(f"  Total proteins: {summary['total_proteins']}")
    print(f"  With PDB structures: {summary['with_pdb']}")
    print(f"  With GO terms: {summary['with_go']}")
    print(f"  Length range: {summary['length_stats']['min']}-{summary['length_stats']['max']} "
          f"(mean: {summary['length_stats']['mean']:.0f})")
    print(f"  Saved to: {output_path}")

    return summary


# --- Helper functions for parsing UniProt JSON ---

def _extract_protein_name(protein: dict) -> str:
    desc = protein.get("proteinDescription", {})
    rec = desc.get("recommendedName", {})
    if rec:
        return rec.get("fullName", {}).get("value", "")
    sub = desc.get("submissionNames", [])
    if sub:
        return sub[0].get("fullName", {}).get("value", "")
    return ""


def _extract_go_terms(protein: dict) -> list[dict]:
    refs = protein.get("uniProtKBCrossReferences", [])
    go_terms = []
    for ref in refs:
        if ref.get("database") == "GO":
            go_id = ref.get("id", "")
            props = {p["key"]: p["value"] for p in ref.get("properties", [])}
            go_terms.append({
                "id": go_id,
                "term": props.get("GoTerm", ""),
                "evidence": props.get("GoEvidenceType", ""),
            })
    return go_terms


def _extract_ec(protein: dict) -> list[str]:
    desc = protein.get("proteinDescription", {})
    rec = desc.get("recommendedName", {})
    ec_numbers = []
    for ec in rec.get("ecNumbers", []):
        ec_numbers.append(ec.get("value", ""))
    return ec_numbers


def _extract_keywords(protein: dict) -> list[str]:
    return [kw.get("name", "") for kw in protein.get("keywords", [])]


def _extract_pdb_ids(protein: dict) -> list[str]:
    refs = protein.get("uniProtKBCrossReferences", [])
    return [ref["id"] for ref in refs if ref.get("database") == "PDB"]


def _extract_xrefs(protein: dict, db_name: str) -> list[str]:
    refs = protein.get("uniProtKBCrossReferences", [])
    return [ref["id"] for ref in refs if ref.get("database") == db_name]


def _extract_features(protein: dict) -> list[dict]:
    """Extract functional site annotations (active sites, binding sites, etc.)."""
    features = []
    for feat in protein.get("features", []):
        feat_type = feat.get("type", "")
        if feat_type in ("Active site", "Binding site", "Metal binding", "Site",
                        "Calcium binding", "Zinc finger"):
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if start is not None:
                features.append({
                    "type": feat_type,
                    "start": start,
                    "end": end if end else start,
                    "description": feat.get("description", ""),
                })
    return features


if __name__ == "__main__":
    curate_pilot_dataset()
