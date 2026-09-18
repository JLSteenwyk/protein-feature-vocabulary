"""Regenerate circuit and steering summaries using explicit copied-input mappings."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from common import ROOT
from probe_checkpoint import atomic_json
from stage_revision_release import digest
from verify_relocated_manifest import records


RUNS = ('circuit_controls/esm2', 'circuit_controls/esm3', 'steering_controls')


def scientific_view(summary):
    result = deepcopy(summary)
    for key in ('identity', 'selection', 'verifier', 'scores_export', 'path_relocations'):
        result.pop(key, None)
    for row in result.get('pairs', {}).values():
        row.pop('effects', None)
    return result


def compare_values(a, b):
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('Scientific field inventory differs')
        return max((compare_values(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError('Scientific list dimensions differ')
        return max((compare_values(x, y) for x, y in zip(a, b)), default=0.)
    if isinstance(a, float):
        if not isinstance(b, (float, int)) or not np.isfinite([a, b]).all() or not np.isclose(a, b, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Scientific numerical mismatch: {a}, {b}')
        return abs(a-b)
    if type(a) is not type(b) or a != b:
        raise ValueError('Scientific non-floating value differs')
    return 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    directory = args.workdir.absolute()
    if directory.is_relative_to(ROOT) or ROOT.is_relative_to(directory):
        raise ValueError('Use an independent new tree outside the project')
    if args.report.exists():
        raise FileExistsError('Use a new report pathname')
    directory.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Full raw-record circuit/steering summary regeneration with explicit copied-input mappings; not foundation-model inference or checkpoint resumption.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'runner': digest(Path(__file__)), 'complete': False,
              'copied_files': {}, 'runs': {}, 'comparison_tolerance': {'rtol': 1e-12, 'atol': 1e-12}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    copied = {}

    def copy(source, target):
        source, target = Path(source), Path(target)
        expected = digest(source)
        if target not in copied:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied[target] = expected
            report['copied_files'][str(target.relative_to(directory))] = expected
        if copied[target] != expected or digest(target) != expected:
            raise ValueError(f'Input copy differs: {source}')
        return expected

    try:
        for base in [ROOT/'src', ROOT/'scripts/revision']:
            for source in sorted(base.rglob('*.py')):
                copy(source, directory/source.relative_to(ROOT))
        mappings = [f'{ROOT}={directory}']
        baseline = {}
        for name in RUNS:
            source_run = ROOT/'revision/analyses'/name
            baseline[name] = json.loads((source_run/'verified_summary.json').read_text())
            identity = json.loads((source_run/'identity.json').read_text())
            for _, record in records(identity):
                source = Path(record['path'])
                if source.is_relative_to(ROOT):
                    target = directory/source.relative_to(ROOT)
                else:
                    target = directory/'external_inputs'/record['sha256']/source.name
                    mapping = f'{source}={target}'
                    if mapping not in mappings:
                        mappings.append(mapping)
                expected = copy(source, target)
                if expected != {k: record[k] for k in ('bytes', 'sha256')}:
                    raise ValueError(f'Historical input content differs: {source}')
            for source in sorted(source_run.rglob('*')):
                if not source.is_file() or source.name.startswith('.'):
                    continue
                if (source.suffix not in ('.json', '.npz') or source.name == 'verified_summary.json'
                        or 'paired_readouts' in source.name or 'paired_effects' in source.name):
                    continue
                copy(source, directory/source.relative_to(ROOT))
        report['mappings'] = mappings
        atomic_json(args.report, report)
        for name in RUNS:
            run = directory/'revision/analyses'/name
            script = 'summarize_steering_controls_relocated.py' if name == 'steering_controls' else 'summarize_circuit_controls_relocated.py'
            command = [str(args.python.absolute()), f'scripts/revision/{script}', '--run-dir', str(run)]
            for mapping in mappings:
                command += ['--map', mapping]
            env = dict(os.environ, PYTHONPATH='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
            result = subprocess.run(command, cwd=directory, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            row = {'command': command, 'exit_code': result.returncode, 'output': result.stdout}
            report['runs'][name] = row
            if result.returncode:
                raise RuntimeError(f'Relocated summary failed: {name}')
            new = json.loads((run/'verified_summary.json').read_text())
            old = baseline[name]
            row['summary_max_absolute_difference'] = compare_values(scientific_view(old), scientific_view(new))
            exports = [('paired_readouts.npz', old['scores_export'])] if name == 'steering_controls' else [
                (f'{pair}_paired_effects.npz', data['effects']) for pair, data in old['pairs'].items()]
            row['arrays'] = {}
            for filename, reference in exports:
                original = Path(reference['path'])
                if digest(original) != {k: reference[k] for k in ('bytes', 'sha256')}:
                    raise ValueError('Archived array export changed')
                with np.load(original, allow_pickle=False) as a, np.load(run/filename, allow_pickle=False) as b:
                    if set(a.files) != set(b.files):
                        raise ValueError('Export array inventory differs')
                    for key in a.files:
                        if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype or not np.allclose(a[key], b[key], rtol=1e-12, atol=1e-12):
                            raise ValueError(f'Export array differs: {name}/{key}')
                    row['arrays'][filename] = {'sha256': digest(run/filename)['sha256'],
                                               'all_arrays_exact': all(np.array_equal(a[k], b[k]) for k in a.files)}
            row['verified'] = True
            atomic_json(args.report, report)
            print(f'{name}: all scientific summaries and arrays reproduced', flush=True)
        for path, expected in copied.items():
            if digest(path) != expected:
                raise ValueError(f'Copied input changed: {path}')
        report['complete'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)


if __name__ == '__main__':
    main()
