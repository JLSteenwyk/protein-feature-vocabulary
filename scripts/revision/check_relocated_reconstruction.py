"""Replay matched reconstruction from copied hidden states and fitted SAEs."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from common import ROOT
from probe_checkpoint import atomic_json
from stage_revision_release import digest


def scientific_summary(value):
    result = copy.deepcopy(value)
    for model in result['models'].values():
        # Regenerated archives have new timestamps; arrays are checked separately.
        model.pop('records')
    return result


def compare_arrays(expected, actual):
    with np.load(expected, allow_pickle=False) as left, np.load(actual, allow_pickle=False) as right:
        if set(left.files) != set(right.files):
            raise ValueError('Reconstruction array inventory differs')
        checked = {}
        for key in left.files:
            a, b = left[key], right[key]
            if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
                raise ValueError(f'Reconstruction array differs: {key}')
            checked[key] = {'shape': list(a.shape), 'dtype': str(a.dtype), 'exact': True}
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    directory = args.workdir.resolve()
    if directory.is_relative_to(ROOT) or ROOT.is_relative_to(directory):
        raise ValueError('Use a new independent tree outside the project')
    if args.report.exists():
        raise FileExistsError('Use a new report pathname')
    directory.mkdir(parents=True, exist_ok=False)
    baseline = ROOT/'revision/analyses/reconstruction_resampled.json'
    original = json.loads(baseline.read_text())
    report = {'scope': 'Full 50,000-position reconstruction and 500 cluster resamples from copied hidden states/checkpoints; not foundation inference or SAE retraining.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'runner': digest(Path(__file__)), 'baseline': digest(baseline),
              'copied_files': {}, 'records': {}, 'complete': False}
    paths = [ROOT/'scripts/revision'/name for name in
             ['recompute_reconstruction.py', 'common.py', 'probe_checkpoint.py']]
    paths += sorted((ROOT/'src').rglob('*.py'))
    paths += [ROOT/'revision/analyses/clusters.json',
              ROOT/'results/scaled_1.5M/sae_features/esm3/residue/protein_summaries.h5']
    for model in ['esm3', 'esm2']:
        chunks = sorted((ROOT/f'results/scaled_1.5M/activations/{model}/residue_L33').glob('chunk*.h5'))
        if not chunks:
            raise ValueError(f'No hidden-state inputs for {model}')
        paths += chunks + [ROOT/f'models/sae_1.5M/{model}_residue_ef8_k64/best.pt']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    try:
        for model, record in original['models'].items():
            path = ROOT/record['records']['path']
            expected = {k: record['records'][k] for k in ['bytes', 'sha256']}
            if digest(path) != expected:
                raise ValueError(f'Archived numerical records changed: {model}')
            report['records'][model] = expected
        for source in paths:
            name = str(source.relative_to(ROOT))
            expected = digest(source)
            target = directory/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != expected or digest(source) != expected:
                raise ValueError(f'Copy/source changed: {name}')
            report['copied_files'][name] = expected
            print('Verified copy:', name, flush=True)
        command = [str(args.python.absolute()), 'scripts/revision/recompute_reconstruction.py',
                   '--n', '50000', '--bootstrap', '500', '--threads', '2']
        report['command'] = command
        atomic_json(args.report, report)
        env = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='',
                   OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (directory/'reconstruction.log').open('w') as log:
            run = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['exit_code'] = run.returncode
        if run.returncode:
            raise RuntimeError('Reconstruction replay failed; see workdir/reconstruction.log')
        reproduced = directory/'revision/analyses/reconstruction_resampled.json'
        current = json.loads(reproduced.read_text())
        report['result'] = digest(reproduced)
        report['summary_exact'] = scientific_summary(original) == scientific_summary(current)
        if not report['summary_exact']:
            raise ValueError('Scientific reconstruction summary differs')
        report['arrays'] = {}
        for model, record in original['models'].items():
            path = ROOT/record['records']['path']
            target = directory/current['models'][model]['records']['path']
            if digest(path) != report['records'][model]:
                raise ValueError('Archived numerical records changed during replay')
            report['arrays'][model] = compare_arrays(path, target)
        if digest(baseline) != report['baseline']:
            raise ValueError('Archived summary changed during replay')
        for name, expected in report['copied_files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Relocated input/code changed: {name}')
        report['complete'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)
    print('Both reconstruction summaries and all numerical arrays reproduce exactly.')


if __name__ == '__main__':
    main()
