"""Run the unchanged training auditor with explicit relocated inputs and new output."""
import argparse
import hashlib
import json
from pathlib import Path

import audit_training_provenance as audit
from probe_checkpoint import atomic_json, file_identity


def run(storage, project_root, analysis_directory, output):
    storage, project_root, analysis_directory, output = [
        Path(p).resolve() for p in (storage, project_root, analysis_directory, output)]
    hits = analysis_directory/'evaluation_vs_training.tsv'
    if not hits.is_file():
        raise FileNotFoundError(hits)
    checkpoints = []
    chunks = {}
    for model in ['esm3', 'esm2']:
        checkpoint = project_root/f'models/sae_1.5M/{model}_residue_ef8_k64/best.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoints.append(checkpoint)
        paths = sorted((storage/model/'residue_L33').glob('*.h5'))
        if not paths:
            raise FileNotFoundError(f'No {model} training chunks under {storage}')
        chunks[model] = [{'path': str(p), 'bytes': p.stat().st_size} for p in paths]
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve a new pathname before hashing or running; never replace archived output.
    with output.open('x') as handle:
        json.dump({'complete': False, 'status': 'initializing'}, handle)
    source_paths = [Path(__file__).resolve(), Path(audit.__file__).resolve()]
    report = {'complete': False, 'status': 'running',
              'scope': 'Relocated invocation of the unchanged current-sampler reconstruction; not proof of historical training-row selection.',
              'storage': str(storage), 'project_root': str(project_root),
              'analysis_directory': str(analysis_directory),
              'sources': [file_identity(p, content=True) for p in source_paths],
              'inputs': [file_identity(p, content=True) for p in [hits, *checkpoints]],
              'chunk_inventory': chunks,
              'chunk_identity_scope': 'Paths and byte counts only; full activation contents are not rehashed by this runner.',
              'audit': None}
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths]
    original = audit.STORAGE, audit.ROOT, audit.OUT, audit.save

    def record(name, result):
        if name != 'training_provenance_audit.json':
            raise ValueError(f'Unexpected audit output: {name}')
        report['audit'] = result
        atomic_json(output, report)

    try:
        audit.STORAGE, audit.ROOT, audit.OUT, audit.save = storage, project_root, analysis_directory, record
        audit.main()
        if before != [hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths]:
            raise ValueError('Auditor or runner source changed during execution')
        report.update(complete=True, status='complete')
    except Exception as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        audit.STORAGE, audit.ROOT, audit.OUT, audit.save = original
        atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', required=True, type=Path,
                        help='Directory containing esm2/residue_L33 and esm3/residue_L33')
    parser.add_argument('--project-root', required=True, type=Path,
                        help='Project tree containing models/sae_1.5M')
    parser.add_argument('--analysis-directory', required=True, type=Path,
                        help='Directory containing evaluation_vs_training.tsv')
    parser.add_argument('--output', required=True, type=Path,
                        help='New JSON pathname; existing files are refused')
    args = parser.parse_args()
    run(args.storage, args.project_root, args.analysis_directory, args.output)


if __name__ == '__main__':
    main()
