"""Record isolated CUDA dependency, regression and cached-inference checks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess

from check_cpu_environment import ROOT, identity
from probe_checkpoint import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--gpu', required=True, type=int)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    executable = str(args.python.absolute())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2')
    code = ('import json,sys,platform,torch; print(json.dumps(dict('
            'python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,'
            'platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,'
            'available=torch.cuda.is_available())))')
    info = json.loads(subprocess.check_output([executable, '-c', code], env=env, text=True))
    if (info['prefix'] == info['base_prefix'] or info['torch'] != '2.6.0+cu124'
            or info['cuda'] != '12.4' or not info['available']):
        raise ValueError('Expected isolated CUDA 12.4 / PyTorch 2.6.0 environment')
    if 'include-system-site-packages = false' not in (Path(info['prefix'])/'pyvenv.cfg').read_text():
        raise ValueError('Environment must not inherit system packages')
    directory = ROOT/'revision/reproducibility'
    lock_handle = (directory/'.gpu_environment.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tests = sorted((ROOT/'tests').glob('test_revision*.py'))
    sources = sorted(set((ROOT/'scripts/revision').glob('*.py')) |
                     set((ROOT/'src').rglob('*.py')) | set(tests))
    pins = [directory/'requirements-analysis.txt', directory/'requirements-inference.txt']
    inventory = [identity(p) for p in sources + pins]
    report = {'scope': 'Isolated Linux GPU environment: offline regression suite and synthetic cached ESM-2/ESM-3 inference only; not full-cohort reproduction or all installed APIs.',
              'environment': info, 'gpu': args.gpu, 'source_inventory': inventory,
              'commands': [], 'complete': False, 'passed': False}
    output = directory/'gpu_environment_check.json'
    atomic_json(output, report)
    commands = [[executable, '-m', 'pip', 'check'],
                [executable, '-m', 'pytest', *[str(p.relative_to(ROOT)) for p in tests], '-q']]
    commands += [[executable, 'scripts/revision/check_clean_inference.py', '--model', name]
                 for name in ['esm2', 'esm3']]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report['commands'].append({'command': command, 'exit_code': result.returncode,
                                   'output': result.stdout})
        atomic_json(output, report)
        print(result.stdout, end='', flush=True)
        if result.returncode:
            raise SystemExit('GPU environment check failed; no lock certified')
    report['runtime_reports'] = []
    for name in ['esm2', 'esm3']:
        path = directory/'gpu_runtime'/f'{name}.json'
        runtime = json.loads(path.read_text())
        if not runtime['passed'] or not runtime['complete']:
            raise ValueError('Incomplete runtime report')
        report['runtime_reports'].append(identity(path))
    # ESM declares this legacy dependency, but the tested inference path does not use it.
    diagnostic = subprocess.run([executable, '-c', 'import torchtext'], env=env,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report['torchtext_diagnostic'] = {'exit_code': diagnostic.returncode,
                                    'output': diagnostic.stdout,
                                    'required_by_tested_inference_path': False}
    frozen = subprocess.check_output([executable, '-m', 'pip', 'freeze'], text=True)
    if any(' @ ' in row or row.startswith(('-e', '/')) for row in frozen.splitlines()):
        raise ValueError('Nonportable direct/local dependency')
    if inventory != [identity(p) for p in sources + pins]:
        raise ValueError('Source or requirement changed during check')
    lock = directory/'requirements-inference-lock.txt'
    lock.write_text('# Tested Linux/Python 3.11 cached ESM inference paths only.\n'
                    '# Install torch/torchvision from the cu124 index first.\n'
                    '# torchtext import is not certified; see INSTALL.md.\n'+frozen)
    report.update(lock=identity(lock), complete=True, passed=True)
    atomic_json(output, report)
    print('GPU environment checks passed within the recorded scope.')


if __name__ == '__main__':
    main()
