"""Inventory and rehash already-copied model-cache inputs; no download or upload."""
import argparse
import json
from pathlib import Path

from probe_checkpoint import atomic_json
from stage_revision_release import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new inventory output')
    source_identity = digest(args.replay_report)
    source = json.loads(args.replay_report.read_text())
    if not source['complete']:
        raise ValueError('Only completed replay inputs may be inventoried')
    root = Path(source['workdir']).resolve()
    files = {name: expected for name, expected in source['copied_files'].items()
             if name.startswith('external_inputs/')}
    if len(files) != 6:
        raise ValueError('Expected the six recorded ESM model-cache inputs')
    for name, expected in files.items():
        target = (root/name).resolve()
        if not target.is_relative_to(root/'external_inputs') or digest(target) != expected:
            raise ValueError(f'External copied input differs: {name}')
        print('Verified', name, flush=True)
    if digest(args.replay_report) != source_identity:
        raise ValueError('Replay report changed during inventory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, {
        'schema': 1, 'path_base': 'project root', 'algorithm': 'sha256',
        'local_root': str(root), 'complete': True, 'publication_ready': False,
        'scope': 'Read-only inventory of six existing copied model-cache files. No new bundle copy, acquisition, license grant, redistribution or publication.',
        'source_replay': {'path': str(args.replay_report.resolve()), **source_identity},
        'script': digest(Path(__file__)), 'files': files,
        'expected_files': len(files), 'total_bytes': sum(r['bytes'] for r in files.values())})


if __name__ == '__main__':
    main()
