"""Regenerate OUP clean/red and supplement PDFs from copied source assets."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from common import ROOT
from stage_revision_release import digest
from probe_checkpoint import atomic_json


def figure_inputs(root):
    root = Path(root).resolve()
    source = (root/'revision/manuscript/sn-article.tex').read_text()
    paths = []
    for image in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source):
        choices = [root/'revision'/parent/image for parent in ['manuscript', 'submitted', 'figures']]
        found = next((p.resolve() for p in choices if p.is_file()), None)
        if found is None or not found.is_relative_to(root):
            raise ValueError(f'Missing or external manuscript figure: {image}')
        paths.append(found)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', required=True, type=Path)
    parser.add_argument('--tectonic', required=True, type=Path)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    directory = args.workdir.absolute()
    if directory.is_relative_to(ROOT) or ROOT.is_relative_to(directory):
        raise ValueError('Use a new independent tree outside the project')
    if args.report.exists():
        raise FileExistsError('Use a new report pathname')
    directory.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Copied-source OUP clean/red and supplement builds; text and raster equality with reviewed artifacts. Uses installed native tools/cache, not a clean TeX installation or new scientific validation.',
              'workdir': str(directory), 'complete': False, 'copied_files': {}, 'commands': [], 'documents': {},
              'runner': digest(Path(__file__)), 'tectonic': {'path': str(args.tectonic.absolute()), **digest(args.tectonic)}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    env = dict(os.environ, PYTHONPATH='', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')

    def execute(command, cwd, log=None):
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report['commands'].append({'command': command, 'cwd': str(cwd),
                                   'exit_code': result.returncode, 'output': result.stdout})
        if log:
            log.write_text(result.stdout)
        atomic_json(args.report, report)
        if result.returncode:
            raise RuntimeError(f'Document command failed: {command}')

    try:
        paths = list((ROOT/'scripts/revision').glob('*.py'))
        for name in ['revision/vendor', 'revision/figures', 'revision/submitted']:
            paths += [p for p in (ROOT/name).rglob('*') if p.is_file() and not p.name.startswith('.')]
        paths += [p for p in (ROOT/'revision/manuscript').iterdir()
                  if p.suffix in {'.tex', '.bib', '.bst', '.cls', '.sty'}]
        paths += figure_inputs(ROOT)
        for source in sorted(set(paths)):
            relative = source.relative_to(ROOT)
            target = directory/relative
            expected = digest(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != expected or digest(source) != expected:
                raise ValueError(f'Copied source changed: {relative}')
            report['copied_files'][str(relative)] = expected
        for name in ['revision/analyses', 'revision/journal', 'revision/supplement']:
            (directory/name).mkdir(parents=True, exist_ok=True)
        python = str(args.python.absolute())
        for generator in ['build_journal_manuscript.py', 'build_supplement.py']:
            execute([python, f'scripts/revision/{generator}'], directory)
        documents = [('revision/journal/bioinformatics', 'clean-engine.log'),
                     ('revision/journal/bioinformatics-marked', 'marked-engine.log'),
                     ('revision/supplement/supplement-clean', 'relocated-engine.log')]
        for stem, log in documents:
            tex = directory/(stem+'.tex')
            execute([str(args.tectonic.absolute()), '-k', '--keep-logs', tex.name], tex.parent, tex.parent/log)
        execute([python, 'scripts/revision/verify_journal_manuscript.py', '--engine-logs',
                 'revision/journal/clean-engine.log', 'revision/journal/marked-engine.log', '--require-length'], directory)
        execute([python, 'scripts/revision/verify_supplement.py'], directory)
        raster = directory/'render_comparison'
        raster.mkdir()
        for stem, _ in documents:
            old, new = ROOT/(stem+'.pdf'), directory/(stem+'.pdf')
            a = subprocess.check_output(['pdftotext', '-layout', str(old), '-'])
            b = subprocess.check_output(['pdftotext', '-layout', str(new), '-'])
            if a != b:
                raise ValueError(f'Rendered text differs: {stem}')
            basename = Path(stem).name
            for label, pdf in [('original', old), ('relocated', new)]:
                execute(['pdftoppm', '-scale-to', '1200', '-png', str(pdf), str(raster/f'{basename}-{label}')], directory)
            originals = sorted(raster.glob(f'{basename}-original-*.png'))
            replicas = sorted(raster.glob(f'{basename}-relocated-*.png'))
            if not originals or len(originals) != len(replicas):
                raise ValueError('Raster page counts differ')
            if any(digest(p) != digest(q) for p, q in zip(originals, replicas)):
                raise ValueError(f'Raster appearance differs: {stem}')
            report['documents'][stem] = {'original': digest(old), 'relocated': digest(new),
                                         'text_exact': True, 'all_page_rasters_exact': True,
                                         'pages': len(originals), 'raster_max_dimension': 1200}
            print(f'{basename}: {len(originals)} pages, exact text and raster equality', flush=True)
        for name, expected in report['copied_files'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Copied source changed during build: {name}')
        report['complete'] = True
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)


if __name__ == '__main__':
    main()
