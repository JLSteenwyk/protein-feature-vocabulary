"""Check an isolated CPU test environment without altering the analysis environment."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def identity(path):
    return {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    args = parser.parse_args()
    executable = str(args.python.absolute())
    code = ('import json,sys,platform,torch; print(json.dumps(dict('
            'python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,'
            'platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda)))')
    info = json.loads(subprocess.check_output([executable, '-c', code], text=True))
    if info['prefix'] == info['base_prefix'] or info['torch'] != '2.6.0+cpu' or info['cuda'] is not None:
        raise ValueError('Expected an isolated venv with PyTorch 2.6.0+cpu')
    cfg = Path(info['prefix'])/'pyvenv.cfg'
    if 'include-system-site-packages = false' not in cfg.read_text():
        raise ValueError('Test environment must not inherit system packages')
    test_paths = sorted((ROOT/'tests').glob('test_revision*.py'))
    commands = [[executable, '-m', 'pip', 'check'],
                [executable, '-m', 'pytest', *[str(p.relative_to(ROOT)) for p in test_paths], '-q']]
    runs = []
    for command in commands:
        run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False)
        runs.append({'command': command, 'exit_code': run.returncode, 'output': run.stdout})
    frozen = subprocess.check_output([executable, '-m', 'pip', 'freeze'], text=True)
    if any(' @ ' in row or row.startswith(('-e', '/')) for row in frozen.splitlines()):
        raise ValueError('Nonportable direct/local dependency in environment')
    directory = ROOT/'revision/reproducibility'
    source_paths = sorted(set((ROOT/'scripts/revision').glob('*.py')) |
                          set((ROOT/'src').rglob('*.py')) |
                          set(test_paths))
    pins = [directory/'requirements-analysis.txt', directory/'requirements-cpu-tests.txt']
    passed = all(run['exit_code'] == 0 for run in runs)
    report = {'scope': 'Clean Linux CPU dependency check and offline regression suite; not GPU/model-inference or full-data reproduction.',
              'environment': info, 'requirements': [identity(p) for p in pins],
              'source_inventory': [identity(p) for p in source_paths],
              'commands': runs, 'passed': passed}
    if passed:
        lock = directory/'requirements-cpu-tests-lock.txt'
        lock.write_text('# Clean-tested Linux/Python 3.11 CPU environment.\n'
                        '# Install torch==2.6.0+cpu from the PyTorch CPU index first.\n'+frozen)
        report['lock'] = identity(lock)
    (directory/'cpu_environment_check.json').write_text(json.dumps(report, indent=2)+'\n')
    for run in runs:
        print(run['output'], end='')
    if not passed:
        sys.exit('Clean CPU environment checks failed; no lock certified')


if __name__ == '__main__':
    main()
