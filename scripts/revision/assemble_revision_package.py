"""Assemble a local revision package with explicit frozen input inventories."""
import argparse
import json
from pathlib import Path
import shutil

from stage_revision_release import ROOT, candidates, digest


def add(plan, name, source, expected=None):
    relative = Path(name)
    source = Path(source)
    if (relative.is_absolute() or '..' in relative.parts or not relative.parts
            or relative.as_posix() == 'revision-package-manifest.json'):
        raise ValueError(f'Unsafe package path: {name}')
    if source.is_symlink() or any(p.is_symlink() for p in source.parents):
        raise ValueError(f'Symlink requires review: {source}')
    current = digest(source)
    if expected is not None and current != expected:
        raise ValueError(f'Frozen input changed: {source}')
    record = {'source': str(source), **current}
    if name in plan and any(plan[name][k] != current[k] for k in current):
        raise ValueError(f'Conflicting package identity: {name}')
    plan.setdefault(name, record)


def assemble(root, destination, bundles):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('Use a new destination outside the source workspace')
    if destination.exists():
        raise FileExistsError(destination)
    plan = {}
    for source in candidates(root):
        add(plan, str(source.relative_to(root)), source)
    add(plan, 'LICENSE', root/'LICENSE')
    # Keep release evidence, but not older ZIPs or nested public-code templates.
    for source in sorted((root/'revision/release').iterdir()):
        if source.is_file() and source.suffix in {'.md', '.json', '.xml', '.sha256'}:
            add(plan, str(source.relative_to(root)), source)
    inventory = []
    for manifest_path, input_root in bundles:
        manifest_path, input_root = Path(manifest_path).resolve(), Path(input_root).resolve()
        if destination.is_relative_to(input_root) or input_root.is_relative_to(destination):
            raise ValueError('Destination overlaps an input bundle')
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('complete') or manifest.get('path_base') != 'project root':
            raise ValueError(f'Incomplete or unsupported inventory: {manifest_path}')
        inventory.append({'manifest': str(manifest_path), 'root': str(input_root),
                          **digest(manifest_path)})
        for name, expected in manifest['files'].items():
            add(plan, name, input_root/name,
                {k: expected[k] for k in ('bytes', 'sha256')})
    destination.mkdir(parents=True)
    report = {'schema': 1, 'path_base': 'project root', 'algorithm': 'sha256',
              'complete': False, 'publication_ready': False,
              'scope': 'Internal joined code/document/result/input package, including author notes and third-party inputs. Not authorized public selection, upstream training replication or a restored model cache.',
              'script': digest(Path(__file__)), 'input_inventories': inventory,
              'remaining': ['Public repository URL and distribution authorization',
                            'Final workflow/dependency and cross-artifact review',
                            'External training-activation store is not bundled'],
              'files': {}, 'sources': {k: v['source'] for k, v in plan.items()}}
    output = destination/'revision-package-manifest.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        for name, record in sorted(plan.items()):
            source, target = Path(record['source']), destination/name
            expected = {k: record[k] for k in ('bytes', 'sha256')}
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(source) != expected or digest(target) != expected:
                raise ValueError(f'Changed source or incorrect copy: {name}')
            report['files'][name] = expected
        report.update(complete=True, expected_files=len(plan),
                      total_bytes=sum(v['bytes'] for v in report['files'].values()))
    except Exception as error:
        report['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        output.write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--bundle', nargs=2, action='append', default=[],
                        metavar=('MANIFEST', 'INPUT_ROOT'))
    args = parser.parse_args()
    report = assemble(args.project_root, args.destination, args.bundle)
    print({k: report[k] for k in ('complete', 'publication_ready', 'expected_files', 'total_bytes')})


if __name__ == '__main__':
    main()
