"""Reproduce the full corrected GO audit in a new tree using copied inputs/code."""
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
        raise ValueError('Use a new independent tree outside the project')
    if args.report.exists():
        raise FileExistsError('Use a new report pathname')
    directory.mkdir(parents=True, exist_ok=False)
    paths = ['scripts/revision/run_go_audit.py', 'scripts/revision/common.py',
             'data/eval_expanded/metadata.json', 'results/scaled_1.5M/decoder_pca.json',
             'results/scaled_1.5M/go_enrichment.json']
    paths += [f'results/scaled_1.5M/sae_features/{model}/residue/protein_summaries.h5'
              for model in ['esm2', 'esm3']]
    baseline = ROOT/'revision/analyses/go_audit.json'
    report = {'scope': 'Full corrected GO audit re-executed from copied code and five inputs; not other analyses or foundation-model re-extraction.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'baseline': {'path': str(baseline), **digest(baseline)},
              'runner': {'path': str(Path(__file__).resolve()), **digest(Path(__file__))},
              'copied_files': {}, 'complete': False, 'matches_archive': False}
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
        command = [str(args.python.absolute()), 'scripts/revision/run_go_audit.py']
        env = dict(os.environ, PYTHONPATH='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        result = subprocess.run(command, cwd=directory, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report['command'] = command
        report['exit_code'] = result.returncode
        report['output'] = result.stdout
        if result.returncode:
            raise RuntimeError('Relocated GO execution failed')
        reproduced = directory/'revision/analyses/go_audit.json'
        report['result'] = {'path': str(reproduced), **digest(reproduced)}
        if digest(baseline) != {k: report['baseline'][k] for k in ('bytes', 'sha256')}:
            raise ValueError('Archive changed during reproduction')
        report['matches_archive'] = json.loads(baseline.read_text()) == json.loads(reproduced.read_text())
        if not report['matches_archive']:
            raise ValueError('Full GO result differs from archived audit')
        for name, expected in report['copied_files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Relocated input/code changed: {name}')
        report['complete'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)
    print('Full relocated GO result matches archived audit exactly.')


if __name__ == '__main__':
    main()
