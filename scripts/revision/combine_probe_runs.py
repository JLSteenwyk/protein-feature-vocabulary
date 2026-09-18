"""Snapshot committed aligned probe continuations without modifying source runs."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from common import OUT, ROOT
from probe_checkpoint import atomic_json


def snapshot_json(path):
    data = path.read_bytes()
    return json.loads(data), {'path': str(path.resolve().relative_to(ROOT)),
                              'sha256': hashlib.sha256(data).hexdigest()}


def validate_selections(base, other):
    if json.loads((base/'probe_split.json').read_text()) != json.loads((other/'probe_split.json').read_text()):
        raise ValueError('Protein split mismatch')
    with np.load(base/'probe_rows.npz') as a, np.load(other/'probe_rows.npz') as b:
        for key in ['train', 'validation', 'test']:
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f'Residue selection mismatch: {key}')


def validate_prediction(reference, prediction):
    for key in ['y', 'protein', 'cluster']:
        if not np.array_equal(reference[key], prediction[key]):
            raise ValueError(f'Prediction alignment mismatch: {key}')
    if prediction['score'].shape != reference['y'].shape or not np.isfinite(prediction['score']).all():
        raise ValueError('Invalid prediction scores')


def combine(base, continuation):
    base = base.resolve(); continuation = continuation.resolve()
    original, identity = snapshot_json(base/'probe_controls.json')
    result = {key: value for key, value in original.items() if key != 'models'}
    result.update(models={}, prediction_files={}, source_snapshots=[identity])
    sources = [(base, original, ['esm3'])]
    if (continuation/'probe_controls.json').exists():
        validate_selections(base, continuation)
        extra, identity = snapshot_json(continuation/'probe_controls.json')
        if extra['sample_counts'] != original['sample_counts'] or extra['seed'] != original['seed']:
            raise ValueError('Sample counts or seed mismatch')
        sources.append((continuation, extra, ['esm2', 'composition']))
        result['source_snapshots'].append(identity)
    composition_path = base/'composition_controls.json'
    if composition_path.exists():
        composition, identity = snapshot_json(composition_path)
        # Prefer the completed original composition run; do not double-count refits.
        sources = [(directory, records, [m for m in models if m != 'composition'])
                   for directory, records, models in sources]
        sources.append((base, {'models': {'composition': composition}}, ['composition']))
        result['source_snapshots'].append(identity)
    elif 'composition' in original['models']:
        sources = [(directory, records, [m for m in models if m != 'composition'])
                   for directory, records, models in sources]
        sources.append((base, original, ['composition']))
    with np.load(base/'probe_predictions_esm3_intact.npz') as reference:
        for directory, records, models in sources:
            for model in models:
                for name, record in records['models'].get(model, {}).items():
                    path = directory/f'probe_predictions_{model}_{name}.npz'
                    before = path.stat()
                    with np.load(path) as prediction:
                        validate_prediction(reference, prediction)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    after = path.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError(f'Prediction changed during snapshot: {path}')
                    result['models'].setdefault(model, {})[name] = record
                    result['prediction_files'].setdefault(model, {})[name] = {
                        'path': str(path.relative_to(ROOT)), 'sha256': digest}
    result['aggregation'] = ('Committed source snapshot; identical canonical residue selections and test metadata verified. '
                             'Original composition controls preferred when available. Source runs unchanged; '
                             'refresh after new fits commit. Historical runs lack fitted-coefficient checkpoints.')
    result['completed_controls'] = sum(map(len, result['models'].values()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run-dir', type=Path, default=OUT)
    parser.add_argument('--continuation-dir', type=Path, default=OUT/'probes_esm2_aligned')
    args = parser.parse_args()
    with (args.base_run_dir/'.combined_probes.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = combine(args.base_run_dir, args.continuation_dir)
        atomic_json(args.base_run_dir/'combined_probe_controls.json', result)
    print(f"Validated {result['completed_controls']}/26 committed controls")


if __name__ == '__main__':
    main()
