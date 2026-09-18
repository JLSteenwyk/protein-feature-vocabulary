"""Replay paired cross-modal SAE encoding from copied hidden states."""
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


def compare_means(expected, actual):
    with np.load(expected, allow_pickle=False) as a, np.load(actual, allow_pickle=False) as b:
        if set(a.files) != {'ids', 's', 'sst'} or set(b.files) != set(a.files):
            raise ValueError('Cross-modal mean array inventory differs')
        result = {}
        for key in a.files:
            left, right = a[key], b[key]
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError(f'Cross-modal mean array differs: {key}')
            if key != 'ids' and not np.isfinite(right).all():
                raise ValueError(f'Nonfinite cross-modal means: {key}')
            result[key] = {'shape': list(left.shape), 'dtype': str(left.dtype), 'exact': True}
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
    paths = [ROOT/'scripts/revision'/name for name in
             ['recompute_cross_modal.py', 'common.py', 'run_go_audit.py']]
    paths += sorted((ROOT/'src').rglob('*.py'))
    paths += [ROOT/'revision/analyses/clusters.json',
              ROOT/'models/sae_1.5M/esm3_residue_ef8_k64/best.pt']
    names = []
    for condition in ['S_only', 'S_St']:
        chunks = sorted((ROOT/f'results/scaled_1.5M/cross_modal/{condition}/residue_L33').glob('*.h5'))
        names.append([p.name for p in chunks])
        paths += chunks
    if names[0] != names[1] or len(names[0]) != 24:
        raise ValueError('Expected all 24 aligned chunk pairs')
    outputs = ['revision/analyses/cross_modal_complete.json',
               'revision/analyses/cross_modal_protein_means.npz']
    report = {'scope': 'Full paired SAE re-encoding, per-feature statistics and 500 cluster resamples from copied hidden states; not foundation-model inference or SAE training.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'runner': digest(Path(__file__)), 'baseline': {p: digest(ROOT/p) for p in outputs},
              'copied_files': {}, 'complete': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    try:
        for source in paths:
            name = str(source.relative_to(ROOT))
            target = directory/name
            expected = digest(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != expected or digest(source) != expected:
                raise ValueError(f'Copy/source changed: {name}')
            report['copied_files'][name] = expected
            atomic_json(args.report, report)
            print('Verified copy:', name, flush=True)
        command = [str(args.python.absolute()), 'scripts/revision/recompute_cross_modal.py',
                   '--threads', '8', '--batch-size', '1024', '--bootstrap', '500']
        report['command'] = command
        env = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='',
                   OPENBLAS_NUM_THREADS='8', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
        with (directory/'cross_modal.log').open('w') as log:
            process = subprocess.Popen(command, cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT)
            report['child_pid'] = process.pid
            atomic_json(args.report, report)
            report['exit_code'] = process.wait()
        if report['exit_code']:
            raise RuntimeError('Cross-modal replay failed; see workdir/cross_modal.log')
        original = json.loads((ROOT/outputs[0]).read_text())
        current = json.loads((directory/outputs[0]).read_text())
        report['summary_exact'] = original == current
        if not report['summary_exact']:
            raise ValueError('Cross-modal per-feature result differs from archive')
        report['mean_arrays'] = compare_means(ROOT/outputs[1], directory/outputs[1])
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
    print('Full cross-modal statistics and protein means reproduce exactly.')


if __name__ == '__main__':
    main()
