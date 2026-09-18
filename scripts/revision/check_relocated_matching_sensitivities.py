"""Replay all matching sensitivities and the bootstrap diagnostic in a new tree."""
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


def compare_arrays(expected, actual):
    with np.load(expected, allow_pickle=False) as a, np.load(actual, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('Sensitivity array inventory differs')
        checked = {}
        for name in a.files:
            left, right = a[name], b[name]
            numeric = np.issubdtype(left.dtype, np.inexact)
            if (left.dtype != right.dtype or left.shape != right.shape or
                    not np.array_equal(left, right, equal_nan=numeric)):
                raise ValueError(f'Sensitivity array differs: {name}')
            checked[name] = {'shape': list(left.shape), 'dtype': str(left.dtype), 'exact': True}
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
    scripts = ['common.py', 'run_matching.py', 'probe_checkpoint.py',
               'diagnose_matching_bootstrap.py', 'matching_sensitivity.py',
               'verify_matching_sensitivity.py']
    paths = ['scripts/revision/'+name for name in scripts]
    paths += ['revision/analyses/'+name for name in
              ['clusters.json', 'training_homology_audit.json', 'matching.json']]
    paths += [f'results/scaled_1.5M/sae_features/{m}/residue/protein_summaries.h5'
              for m in ['esm2', 'esm3']]
    base = ROOT/'revision/analyses'
    outputs = [base/'matching_bootstrap_diagnostic.json', base/'matching_discovery_pairs.npz',
               base/'matching_sensitivity_verification.json', base/'matching_sensitivity/summary.json']
    outputs += sorted((base/'matching_sensitivity').glob('protein_*.json'))
    outputs += sorted((base/'matching_sensitivity').glob('protein_*.npz'))
    if len(outputs) != 28:
        raise ValueError('Expected complete 12-run archived sensitivity suite')
    outputs = [str(path.relative_to(ROOT)) for path in outputs]
    report = {'scope': 'Full bootstrap diagnostic and 12 matching sensitivity reruns; no foundation inference or training-homology search replay.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'runner': digest(Path(__file__)), 'baseline': {p: digest(ROOT/p) for p in outputs},
              'copied_files': {}, 'runs': [], 'comparisons': {}, 'complete': False}
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
        env = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='',
                   OPENBLAS_NUM_THREADS='8', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        report['thread_limit'] = 8
        for script in scripts[-3:]:
            command = [str(args.python.absolute()), 'scripts/revision/'+script]
            log_path = directory/(script+'.log')
            print('Running', script, flush=True)
            with log_path.open('w') as log:
                run = subprocess.run(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
            report['runs'].append({'command': command, 'exit_code': run.returncode, 'log': str(log_path)})
            atomic_json(args.report, report)
            if run.returncode:
                raise RuntimeError(f'Relocated execution failed: {script}')
        for name in outputs:
            source, target = ROOT/name, directory/name
            if source.suffix == '.npz':
                report['comparisons'][name] = {'arrays': compare_arrays(source, target)}
            else:
                if json.loads(source.read_text()) != json.loads(target.read_text()):
                    raise ValueError(f'Sensitivity JSON differs: {name}')
                report['comparisons'][name] = {'json_exact': True}
            report['comparisons'][name]['output'] = digest(target)
            if digest(source) != report['baseline'][name]:
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
    print('All 12 sensitivity runs and bootstrap diagnostic reproduce exactly.')


if __name__ == '__main__':
    main()
