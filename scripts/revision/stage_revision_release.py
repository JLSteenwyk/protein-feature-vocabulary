"""Build a local, checksum-bound revision candidate; never publish it."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
TREES = ('src', 'scripts', 'configs', 'tests', 'revision')
SKIP_PARTS = {'__pycache__', '.pytest_cache', 'release', 'mmseqs_tmp',
              'training_homology_tmp', 'cross_modal_cache',
              'steering_controls_smoke', 'circuit_controls_smoke'}
SKIP_SUFFIXES = {'.pyc', '.lock', '.tmp', '.aux', '.out', '.blg'}


def selected(relative):
    if any(part.startswith('.') or part in SKIP_PARTS for part in relative.parts):
        return False
    return relative.suffix not in SKIP_SUFFIXES


def digest(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'File changed during hashing: {path}')
    return {'bytes': after.st_size, 'sha256': h.hexdigest()}


def candidates(root):
    paths = []
    for tree in TREES:
        directory = root/tree
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        # Do not traverse symlink directories or silently ship external content.
        for path in directory.rglob('*'):
            relative = path.relative_to(root)
            if not selected(relative):
                continue
            if path.is_symlink():
                raise ValueError(f'Symlink requires explicit release review: {relative}')
            if path.is_file():
                paths.append(path)
    paths.extend(root/name for name in ('action_list.txt', 'requirements.txt'))
    return sorted(paths)


def stage(root, destination):
    root, destination = Path(root).resolve(), Path(destination).absolute()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('Stage in a separate tree outside the source workspace')
    paths = candidates(root)
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 1, 'algorithm': 'sha256', 'path_base': 'project root',
                'complete': False, 'publication_ready': False,
                'scope': 'Local code/manuscript/corrected-record candidate only; not complete scientific input closure, a license grant, or authorization to publish.',
                'excluded_directories': sorted(SKIP_PARTS),
                'excluded_suffixes': sorted(SKIP_SUFFIXES),
                'external_inputs': 'See revision/reproducibility/input_manifest.json and revision/reproduction.md; data, models, original results, model caches and external training chunks are not copied here.',
                'remaining': ['Author repository URL and license decision',
                              'Input dependency closure and distribution rights review',
                              'Full relocated scientific execution and final package review'],
                'files': {}}
    manifest_path = destination/'release-candidate-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    counts, sizes = Counter(), Counter()
    for path in paths:
        relative = path.relative_to(root)
        expected = digest(path)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if digest(target) != expected or digest(path) != expected:
            raise ValueError(f'Changed source or incorrect copy: {relative}')
        manifest['files'][str(relative)] = expected
        group = '/'.join(relative.parts[:2]) if relative.parts[0] == 'revision' else relative.parts[0]
        counts[group] += 1
        sizes[group] += expected['bytes']
    if paths != candidates(root):
        raise ValueError('Source file membership changed during staging')
    manifest.update(complete=True, expected_files=len(paths),
                    total_bytes=sum(sizes.values()),
                    groups={key: {'files': counts[key], 'bytes': sizes[key]} for key in sorted(counts)})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--destination', type=Path, required=True,
                        help='New local directory outside source tree; never an existing directory')
    args = parser.parse_args()
    result = stage(args.project_root, args.destination)
    print(json.dumps({key: result[key] for key in ('complete', 'publication_ready', 'expected_files', 'total_bytes', 'groups')}, indent=2))


if __name__ == '__main__':
    main()
