"""Build and verify a local editor-review archive; never submit or publish."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from common import ROOT
from probe_checkpoint import atomic_json
from stage_revision_release import digest


def source_inputs(root):
    revision = Path(root)/'revision'
    names = ['journal/'+name for name in ['bioinformatics.tex', 'bioinformatics-marked.tex',
             'oup-authoring-template.cls', 'oup-abbrvnat.bst', 'oup-local-compat.sty']]
    names.append('manuscript/sn-bibliography.bib')
    text = (revision/'journal/bioinformatics.tex').read_text()
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text)
    if len(figures) != 4:
        raise ValueError('Expected four main figure assets')
    for name in figures:
        if Path(name).name != name or Path(name).suffix not in {'.png', '.pdf', '.eps'}:
            raise ValueError('Figure must have a plain local filename')
        candidates = [revision/parent/name for parent in ['journal', 'submitted', 'figures']]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None or found.is_symlink():
            raise ValueError(f'Missing or linked main figure: {name}')
        names.append(str(found.relative_to(revision)))
    return {name: revision/name for name in sorted(set(names))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--tectonic', required=True, type=Path)
    args = parser.parse_args()
    directory = args.workdir.resolve()
    if directory.is_relative_to(ROOT) or ROOT.is_relative_to(directory):
        raise ValueError('Use a new independent tree outside the project')
    if args.archive.exists() or args.report.exists():
        raise FileExistsError('Use new archive and report paths')
    directory.mkdir(parents=True, exist_ok=False)
    paths = source_inputs(ROOT)
    paths.update({'response_to_reviewers.docx': ROOT/'revision/response_to_reviewers.docx',
                  'supplement-clean.pdf': ROOT/'revision/supplement/supplement-clean.pdf'})
    report = {'scope': 'Local editor-review draft, not authorized submission. Main sources rebuilt using installed native tools/cache; supplement and response copied without source substitution.',
              'complete': False, 'submission_ready': False, 'workdir': str(directory),
              'script': digest(Path(__file__)), 'sources': {}, 'documents': {}, 'commands': []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)

    def execute(command, cwd):
        result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        report['commands'].append({'command': command, 'exit_code': result.returncode, 'output': result.stdout})
        atomic_json(args.report, report)
        if result.returncode:
            raise RuntimeError(f'Editor draft command failed: {command}')

    try:
        for name, source in paths.items():
            expected = digest(source)
            target = directory/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != expected or digest(source) != expected:
                raise ValueError(f'Copied source changed: {name}')
            report['sources'][name] = {'original': str(source), **expected}
        raster = directory/'render_comparison'
        raster.mkdir()
        members = set(paths)
        for stem in ['bioinformatics', 'bioinformatics-marked']:
            execute([str(args.tectonic.absolute()), '-k', '--keep-logs', stem+'.tex'], directory/'journal')
            original = ROOT/f'revision/journal/{stem}.pdf'
            target = directory/f'journal/{stem}.pdf'
            if subprocess.check_output(['pdftotext', '-layout', str(original), '-']) != subprocess.check_output(['pdftotext', '-layout', str(target), '-']):
                raise ValueError(f'Rendered text differs: {stem}')
            for label, pdf in [('original', original), ('draft', target)]:
                execute(['pdftoppm', '-scale-to', '1200', '-png', str(pdf), str(raster/f'{stem}-{label}')], directory)
            left = sorted(raster.glob(stem+'-original-*.png'))
            right = sorted(raster.glob(stem+'-draft-*.png'))
            if len(left) != 7 or len(right) != 7 or any(digest(a) != digest(b) for a, b in zip(left, right)):
                raise ValueError(f'Page count or rendered pages differ: {stem}')
            report['documents'][stem] = {'original': digest(original), 'draft': digest(target),
                                        'pages': 7, 'text_exact': True, 'all_page_rasters_exact': True}
            members.add('journal/'+stem+'.pdf')
        readme = directory/'README.txt'
        readme.write_text('EDITOR-REVIEW DRAFT: NOT FOR SUBMISSION\n\n'
            'Working-revision notices and provisional availability statements remain.\n'
            'Author approval, repository URL/license, revised archival availability and\n'
            'final scientific/release checks are still required. No upload is authorized.\n\n'
            'Main editable sources: journal/bioinformatics.tex and the red wrapper\n'
            'journal/bioinformatics-marked.tex. Build from the journal directory,\n'
            'preserving the adjacent figures and manuscript bibliography directories.\n'
            'The OUP class, bibliography style and local compatibility file are included.\n'
            'Both sources rebuild to seven pages with exact text and 1200-pixel page\n'
            'raster equality under the installed Tectonic/Poppler toolchain. This is\n'
            'not certification of a fresh native-tool installation.\n\n'
            'The clean supplement is PDF only; supplementary LaTeX is intentionally\n'
            'excluded. The response letter is Word. All included files have checksums.\n'
            'No raw records, code release, model weights or author action lists are included.\n')
        members.add('README.txt')
        for name, expected in report['sources'].items():
            identity = {k: expected[k] for k in ['bytes', 'sha256']}
            if digest(directory/name) != identity or digest(Path(expected['original'])) != identity:
                raise ValueError(f'Source changed during build: {name}')
        manifest = {'schema': 1, 'path_base': 'project root', 'complete': True, 'submission_ready': False,
                    'files': {name: digest(directory/name) for name in sorted(members)}}
        atomic_json(directory/'editor-draft-manifest.json', manifest)
        members.add('editor-draft-manifest.json')
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(members):
                archive.write(directory/name, name)
        with zipfile.ZipFile(args.archive) as archive:
            if set(archive.namelist()) != members or len(archive.namelist()) != len(members):
                raise ValueError('Archive membership differs')
            for name in members:
                if hashlib.sha256(archive.read(name)).hexdigest() != digest(directory/name)['sha256']:
                    raise ValueError(f'Archive content differs: {name}')
        report.update(complete=True, archive={'path': str(args.archive.resolve()), **digest(args.archive)},
                      manifest=digest(directory/'editor-draft-manifest.json'), archived_files=len(members))
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        atomic_json(args.report, report)
    print(f'Editor-review archive verified: {len(members)} files; not submission-ready.')


if __name__ == '__main__':
    main()
