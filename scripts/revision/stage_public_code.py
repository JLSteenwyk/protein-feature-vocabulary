"""Stage a minimal local public-code review draft; never initialize Git or publish."""
import argparse
import json
from pathlib import Path
import shutil

from stage_revision_release import digest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    'revision/manuscript/sn-article.tex',
    'revision/manuscript/supplementary-methods.tex',
    'revision/submitted/sn-article.tex',
    'revision/analyses/circuit_controls/esm2/verified_summary.json',
    'revision/analyses/circuit_controls/esm3/verified_summary.json',
    'revision/analyses/steering_controls/verified_summary.json',
)
DEPENDENCIES = ('requirements-analysis.txt', 'requirements-cpu-tests.txt',
                'requirements-cpu-tests-lock.txt', 'requirements-inference.txt',
                'requirements-inference-lock.txt')


def selection(root):
    root = Path(root)
    selected = {}
    selected['LICENSE'] = root/'LICENSE'
    for tree in ['src', 'scripts', 'tests']:
        for source in sorted((root/tree).rglob('*.py')):
            relative = source.relative_to(root)
            if any(p.startswith('.') or p == '__pycache__' for p in relative.parts):
                continue
            selected[str(relative)] = source
    for name in FIXTURES:
        selected[name] = root/name
    for name in DEPENDENCIES:
        relative = 'revision/reproducibility/'+name
        selected[relative] = root/relative
    for name in ['README.md', '.gitignore']:
        selected[name] = root/'revision/release/public_code'/name
    for name, source in selected.items():
        if source.is_symlink() or any(p.is_symlink() for p in source.parents if p != root):
            raise ValueError(f'Symlink requires explicit review: {name}')
        if not source.is_file():
            raise FileNotFoundError(source)
    return dict(sorted(selected.items()))


def stage(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('Use a separate new destination outside the project')
    selected = selection(root)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 1, 'path_base': 'project root', 'algorithm': 'sha256',
                'complete': False, 'publication_ready': False,
                'scope': 'Local code-only author-review selection, not complete scientific input/result distribution, a license grant or publication authorization.',
                'files': {}, 'source_paths': {},
                'license': 'MIT',
                'pending': ['Author repository URL', 'Publication/privacy/third-party-rights review',
                            'Final scientific input/result package and acquisition instructions']}
    manifest_path = destination/'public-code-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    for name, source in selected.items():
        expected = digest(source)
        if expected['bytes'] > 50*1024*1024:
            raise ValueError(f'File exceeds this code-only staging limit: {name}')
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(target) != expected or digest(source) != expected:
            raise ValueError(f'Copy or source changed: {name}')
        manifest['files'][name] = expected
        manifest['source_paths'][name] = str(source.relative_to(root))
    if selection(root) != selected:
        raise ValueError('Source inventory changed during staging')
    manifest.update(complete=True, expected_files=len(selected),
                    total_bytes=sum(v['bytes'] for v in manifest['files'].values()))
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    report = stage(args.project_root, args.destination)
    print({k: report[k] for k in ['complete', 'publication_ready', 'expected_files', 'total_bytes']})


if __name__ == '__main__':
    main()
