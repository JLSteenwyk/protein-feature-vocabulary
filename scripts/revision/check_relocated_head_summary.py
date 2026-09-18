"""Recompute all corrected head summaries from independently copied raw logits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from common import ROOT
from stage_revision_release import digest
from probe_checkpoint import atomic_json


def compare_summaries(original, regenerated, new_root):
    if set(original) != set(regenerated):
        raise ValueError('Summary field inventory changed')
    if {k: v for k, v in original.items() if k != 'inputs'} != {
            k: v for k, v in regenerated.items() if k != 'inputs'}:
        raise ValueError('Scientific summaries differ')
    if len(original['inputs']) != len(regenerated['inputs']):
        raise ValueError('Summary input count differs')
    for old, new in zip(original['inputs'], regenerated['inputs']):
        if any(old[key] != new[key] for key in ('bytes', 'sha256')):
            raise ValueError('Summary input contents differ')
        if not Path(new['path']).is_relative_to(new_root):
            raise ValueError('Regenerated summary reads outside relocated root')


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
    code = ['summarize_corrected_heads.py', 'run_corrected_head_evaluation.py',
            'common.py', 'probe_checkpoint.py', 'capture_reproducibility.py',
            'secondary_structure.py']
    paths = [Path('scripts/revision')/name for name in code]
    paths += [Path('src/models')/name for name in ['esm3_hooks.py', 'interventions.py']]
    run = Path('revision/analyses/corrected_head_evaluation')
    paths += [run/name for name in ['identity.json', 'progress.json']]
    paths += [p.relative_to(ROOT) for p in sorted((ROOT/run).glob('*.npz'))]
    baseline = ROOT/run/'verified_summary.json'
    report = {'scope': 'Full corrected-head raw-logit verification and summary/interval regeneration after relocation; not rerunning model inference, validating biological accuracy or resuming historical checkpoints.',
              'workdir': str(directory), 'python': str(args.python.absolute()),
              'baseline': {'path': str(baseline), **digest(baseline)},
              'runner': {'path': str(Path(__file__).resolve()), **digest(Path(__file__))},
              'copied_files': {}, 'complete': False, 'scientific_fields_equal': False,
              'comparison_rule': 'All fields exactly equal except inputs: verify byte/hash equality and require new-root paths; modification times need not match.'}
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
            report['copied_files'][str(name)] = expected
        command = [str(args.python.absolute()), 'scripts/revision/summarize_corrected_heads.py']
        env = dict(os.environ, PYTHONPATH='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
        result = subprocess.run(command, cwd=directory, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report.update(command=command, exit_code=result.returncode, output=result.stdout)
        if result.returncode:
            raise RuntimeError('Relocated head summarization failed')
        reproduced = directory/run/'verified_summary.json'
        report['result'] = {'path': str(reproduced), **digest(reproduced)}
        if digest(baseline) != {k: report['baseline'][k] for k in ('bytes', 'sha256')}:
            raise ValueError('Archive changed during reproduction')
        compare_summaries(json.loads(baseline.read_text()), json.loads(reproduced.read_text()), directory)
        for name, expected in report['copied_files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Relocated input/code changed: {name}')
        report.update(complete=True, scientific_fields_equal=True)
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)
    print('All relocated head summaries and intervals match the archived scientific fields.')


if __name__ == '__main__':
    main()
