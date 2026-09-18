"""Replay the full matching analysis from copied protein summaries and clusters."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from common import ROOT
from probe_checkpoint import atomic_json
from stage_revision_release import digest


def compare_pairs(expected, actual):
    with np.load(expected, allow_pickle=False) as a, np.load(actual, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('Matching pair array inventory differs')
        result = {}
        for name in a.files:
            left, right = a[name], b[name]
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError(f'Matching pair array differs: {name}')
            result[name] = {'shape': list(left.shape), 'dtype': str(left.dtype), 'exact': True}
    return result


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
    paths = ['scripts/revision/run_matching.py', 'scripts/revision/common.py',
             'revision/analyses/clusters.json']
    paths += [f'results/scaled_1.5M/sae_features/{model}/residue/protein_summaries.h5'
              for model in ['esm2', 'esm3']]
    outputs = ['revision/analyses/matching.json', 'revision/analyses/matching_pairs.npz']
    report = {'scope': 'Full main matching replay, including 500 bootstrap resamples and 99 permutations; not separate split/prevalence sensitivities or foundation inference.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'runner': digest(Path(__file__)), 'baseline': {p: digest(ROOT/p) for p in outputs},
              'copied_files': {}, 'complete': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    try:
        for name in paths:
            source, target = ROOT/name, directory/name
            expected = digest(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != expected or digest(source) != expected:
                raise ValueError(f'Copy/source changed: {name}')
            report['copied_files'][name] = expected
            print('Verified copy:', name, flush=True)
        command = [str(args.python.absolute()), 'scripts/revision/run_matching.py',
                   '--permutations', '99', '--null-features', '512']
        report['command'] = command
        report['thread_limit'] = 8
        atomic_json(args.report, report)
        env = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='',
                   OPENBLAS_NUM_THREADS='8', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        with (directory/'matching.log').open('w') as log:
            run = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
        report['exit_code'] = run.returncode
        if run.returncode:
            raise RuntimeError('Matching replay failed; see workdir/matching.log')
        original = json.loads((ROOT/outputs[0]).read_text())
        current = json.loads((directory/outputs[0]).read_text())
        report['summary_exact'] = original == current
        if not report['summary_exact']:
            raise ValueError('Full matching summary differs from archive')
        report['pair_arrays'] = compare_pairs(ROOT/outputs[1], directory/outputs[1])
        report['outputs'] = {p: digest(directory/p) for p in outputs}
        for name, expected in report['baseline'].items():
            if digest(ROOT/name) != expected:
                raise ValueError(f'Archive changed during replay: {name}')
        for name, expected in report['copied_files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Relocated input/code changed: {name}')
        report['complete'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)
    print('Full matching summary and pair arrays reproduce exactly.')


if __name__ == '__main__':
    main()
