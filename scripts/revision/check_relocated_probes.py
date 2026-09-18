"""Recompute complete paired probe comparisons from copied held-out predictions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from common import ROOT
from stage_revision_release import digest
from probe_checkpoint import atomic_json


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
    report = {'scope': 'Both full paired probe-comparison sets regenerated from copied predictions; not probe refitting, feature extraction, or uncertainty over training.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'runner': digest(Path(__file__)), 'complete': False, 'copied_files': {}, 'cohorts': {}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)

    def copy(relative):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected project-relative snapshot input')
        source, target = ROOT/relative, directory/relative
        expected = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != expected or digest(source) != expected:
            raise ValueError(f'Input copy mismatch: {relative}')
        report['copied_files'][str(relative)] = expected
        return expected

    try:
        for name in ['common.py', 'probe_checkpoint.py', 'paired_probe_comparisons.py']:
            copy(Path('scripts/revision')/name)
        for cohort, relative in [('primary', 'revision/analyses'),
                                 ('homolog_excluded', 'revision/analyses/probes_overlap_excluded')]:
            relative = Path(relative)
            snapshot_path = relative/'combined_probe_controls.json'
            copy(snapshot_path)
            snapshot = json.loads((directory/snapshot_path).read_text())
            for conditions in snapshot['prediction_files'].values():
                for record in conditions.values():
                    if copy(record['path'])['sha256'] != record['sha256']:
                        raise ValueError('Prediction differs from committed snapshot')
            baseline = ROOT/relative/'combined_paired_probe_comparisons.json'
            row = {'baseline': {'path': str(baseline), **digest(baseline)}, 'complete': False}
            report['cohorts'][cohort] = row
            atomic_json(args.report, report)
            command = [str(args.python.absolute()), 'scripts/revision/paired_probe_comparisons.py',
                       '--combined', '--run-dir', str(directory/relative), '--resamples', '500']
            result = subprocess.run(command, cwd=directory, text=True,
                                    env=dict(os.environ, PYTHONPATH='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2'),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            row.update(command=command, exit_code=result.returncode, output=result.stdout)
            if result.returncode:
                raise RuntimeError(f'Relocated paired comparison failed: {cohort}')
            reproduced = directory/relative/'combined_paired_probe_comparisons.json'
            row['result'] = {'path': str(reproduced), **digest(reproduced)}
            old, new = [json.loads(p.read_text()) for p in [baseline, reproduced]]
            if digest(baseline) != {k: row['baseline'][k] for k in ('bytes', 'sha256')}:
                raise ValueError('Baseline changed during check')
            if old != new:
                raise ValueError('Paired comparison result differs from archive')
            row.update(complete=True, exact_equality=True, n_comparisons=len(new))
            atomic_json(args.report, report)
            print(f'{cohort}: all {len(new)} paired comparisons exactly reproduced', flush=True)
        for name, expected in report['copied_files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Copied input changed: {name}')
        report['complete'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)


if __name__ == '__main__':
    main()
