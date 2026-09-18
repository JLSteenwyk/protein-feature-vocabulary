"""Resumable copied-input refits of all 52 controls in both manuscript cohorts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess

from common import ROOT
from probe_checkpoint import atomic_json
from stage_revision_release import digest


COHORTS = {'primary': 'revision/analyses',
           'homolog_excluded': 'revision/analyses/probes_overlap_excluded'}
MODELS = ['esm3', 'esm2', 'composition']


def checked_copy(source, target, expected):
    if source.is_symlink() or target.is_symlink():
        raise ValueError('Symlink input or destination is not supported')
    if digest(source) != expected:
        raise ValueError(f'Source changed: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(target) != expected:
            raise ValueError(f'Existing copy changed: {target}')
        return
    temporary = target.with_name(target.name+'.partial')
    shutil.copy2(source, temporary)
    if digest(temporary) != expected or digest(source) != expected:
        raise ValueError(f'Input changed or copy corrupt: {source}')
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    directory = args.workdir.resolve()
    if directory.is_relative_to(ROOT) or ROOT.is_relative_to(directory):
        raise ValueError('Use an independent tree outside the project')
    if args.report.exists() != args.resume:
        raise ValueError('Existing reports require --resume; new runs require a new report')
    directory.mkdir(parents=True, exist_ok=args.resume)
    with (directory/'.replay.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = {'scripts/revision/'+name for name in [
            'run_portable_probe_refits.py', 'verify_portable_probe_refits.py',
            'common.py', 'probe_checkpoint.py', 'run_probe_controls.py',
            'run_aligned_esm2_probes.py', 'stage_revision_release.py']}
        paths |= {'data/eval_expanded/metadata.json', 'data/eval_expanded/sequences.json',
                  'revision/analyses/clusters.json'}
        paths |= {f'results/scaled_1.5M/sae_features/{model}/residue/{name}'
                  for model in ['esm2', 'esm3'] for name in ['protein_summaries.h5', 'features_sparse.npz']}
        for base in COHORTS.values():
            paths |= {base+'/'+name for name in ['probe_rows.npz', 'probe_split.json', 'combined_probe_controls.json']}
            snapshot = json.loads((ROOT/base/'combined_probe_controls.json').read_text())
            for model in MODELS:
                for record in snapshot['prediction_files'][model].values():
                    name = record['path']
                    path = Path(name)
                    if path.is_absolute() or '..' in path.parts:
                        raise ValueError('Unsafe archived prediction path')
                    if digest(ROOT/name)['sha256'] != record['sha256']:
                        raise ValueError(f'Archived prediction changed: {name}')
                    paths.add(name)
        identity = {'runner': digest(Path(__file__)), 'python': str(args.python.absolute()),
                    'source_root': str(ROOT), 'workdir': str(directory),
                    'inputs': {name: digest(ROOT/name) for name in sorted(paths)}}
        if args.resume:
            report = json.loads(args.report.read_text())
            if report['identity'] != identity:
                raise ValueError('Resume source/code/input identity changed')
        else:
            report = {'scope': 'All controls refitted from copied features, labels and saved splits; reference predictions used only for comparison. Not feature extraction, new sampling or SAE training.',
                      'identity': identity, 'copied_files': {}, 'runs': {}, 'complete': False}
        report['complete'] = False
        report.pop('failure', None)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)
        try:
            for name, expected in identity['inputs'].items():
                checked_copy(ROOT/name, directory/name, expected)
                report['copied_files'][name] = expected
                atomic_json(args.report, report)
            env = dict(os.environ, PYTHONPATH='', CUDA_VISIBLE_DEVICES='',
                       OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            for cohort, base in COHORTS.items():
                for model in MODELS:
                    key = cohort+'/'+model
                    output = 'revision/refits/'+key
                    record = report['runs'].setdefault(key, {})
                    for phase, script in [('fit', 'run_portable_probe_refits.py'),
                                          ('verify', 'verify_portable_probe_refits.py')]:
                        # Refit checkpoints independently validate completed controls on resume.
                        command = [identity['python'], 'scripts/revision/'+script, '--model', model,
                                   '--base-run-dir', base, '--run-dir', output]
                        log_path = directory/(cohort+'_'+model+'_'+phase+'.log')
                        record[phase] = {'command': command, 'log': str(log_path), 'complete': False}
                        print(key, phase, flush=True)
                        with log_path.open('a') as log:
                            process = subprocess.Popen(command, cwd=directory, env=env,
                                stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
                            record[phase]['child_pid'] = process.pid
                            atomic_json(args.report, report)
                            code = process.wait()
                        record[phase].update(exit_code=code, complete=code == 0)
                        atomic_json(args.report, report)
                        if code:
                            raise RuntimeError(f'{key} {phase} failed; inspect {log_path}')
                    verified = directory/output/'refit_verification.json'
                    check = json.loads(verified.read_text())
                    if not check['complete'] or len(check['checks']) != (2 if model == 'composition' else 12):
                        raise ValueError(f'Incomplete refit verification: {key}')
                    record['verification'] = {'path': str(verified), **digest(verified)}
                    atomic_json(args.report, report)
            for name, expected in identity['inputs'].items():
                if digest(ROOT/name) != expected or digest(directory/name) != expected:
                    raise ValueError(f'Source or copied input changed during refits: {name}')
            report['complete'] = True
        except Exception as error:
            report['failure'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            atomic_json(args.report, report)
    print('All 52 relocated refits reproduce metrics and test predictions exactly.')


if __name__ == '__main__':
    main()
