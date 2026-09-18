"""Atomic, input-bound checkpoints for long CPU probe experiments."""
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_identity(path, content=False):
    path = Path(path).resolve()
    stat = path.stat()
    record = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    if content:
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return record


class ProbeCheckpoint:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / '.probe.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = self.directory / 'probe_run_identity.json'
            if manifest.exists():
                if json.loads(manifest.read_text()) != identity:
                    raise ValueError('Checkpoint inputs/protocol changed; use a new run directory.')
            elif (self.directory / 'probe_controls.json').exists():
                raise ValueError('Legacy checkpoint lacks input identity; do not overwrite it.')
            else:
                atomic_json(manifest, identity)
        except BaseException:
            self.lock.close()
            raise

    def load(self, initial):
        path = self.directory / 'probe_controls.json'
        return json.loads(path.read_text()) if path.exists() else initial

    def reusable(self, output, model, name, expected):
        if name not in output.get('models', {}).get(model, {}):
            return False
        path = self.directory / f'probe_predictions_{model}_{name}.npz'
        fit_path = self.directory / f'probe_fit_{model}_{name}.npz'
        if not path.exists() or not fit_path.exists():
            return False
        with np.load(fit_path, allow_pickle=False) as fitted:
            if not np.isfinite(fitted['coef']).all() or not np.isfinite(fitted['intercept']).all():
                raise ValueError(f'Invalid checkpoint coefficients: {model}/{name}')
        with np.load(path, allow_pickle=False) as saved:
            for key, value in expected.items():
                if not np.array_equal(saved[key], value):
                    raise ValueError(f'Checkpoint prediction alignment differs: {model}/{name}/{key}')
            if saved['score'].shape != expected['y'].shape or not np.isfinite(saved['score']).all():
                raise ValueError(f'Invalid checkpoint predictions: {model}/{name}')
        return True

    def commit(self, output, model, name, prediction, fitted):
        for prefix, values in [('probe_predictions', prediction), ('probe_fit', fitted)]:
            target = self.directory / f'{prefix}_{model}_{name}.npz'
            temporary = target.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **values)
            temporary.replace(target)
        # Publish the result only after both associated arrays are complete.
        atomic_json(self.directory / 'probe_controls.json', output)

    def close(self):
        self.lock.close()
