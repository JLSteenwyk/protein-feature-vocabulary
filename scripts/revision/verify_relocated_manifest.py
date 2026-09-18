"""Read-only checksum verification after explicit relocation; never rewrites provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    before = path.stat()
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            sha.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'File changed while hashing: {path}')
    return {'bytes': after.st_size, 'sha256': sha.hexdigest()}


def records(value, pointer='$'):
    if isinstance(value, dict):
        if value.get('schema') == 1 and value.get('path_base') == 'project root' and isinstance(value.get('files'), dict):
            for name, row in value['files'].items():
                yield pointer + '/files/' + name, dict(row, path=name)
            return
        if 'path' in value and 'sha256' in value:
            yield pointer, value
        for key, child in value.items():
            yield from records(child, pointer + '/' + str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from records(child, pointer + '/' + str(index))


def relocate(name, root, mappings):
    if not isinstance(name, str) or not name:
        raise ValueError('Missing file path')
    path = Path(name)
    if '..' in path.parts:
        raise ValueError('Parent traversal is not a supported manifest path')
    if not path.is_absolute():
        return root / path
    for old, new in sorted(mappings, key=lambda item: len(item[0].parts), reverse=True):
        if path.is_relative_to(old):
            return new / path.relative_to(old)
    raise ValueError(f'Absolute path requires an explicit prefix mapping: {name}')


def verify_row(row, root, mappings):
    if not isinstance(row.get('bytes'), int) or isinstance(row['bytes'], bool) or row['bytes'] < 0:
        raise ValueError('Missing or invalid expected byte count')
    if not isinstance(row.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
        raise ValueError('Missing or invalid expected SHA-256')
    target = relocate(row.get('path'), root, mappings)
    if not target.is_file():
        raise ValueError(f'Missing file: {target}')
    if target.stat().st_size != row['bytes']:
        raise ValueError(f'Byte-count mismatch: {target}')
    actual = digest(target)
    if actual['sha256'] != row['sha256']:
        raise ValueError(f'SHA-256 mismatch: {target}')
    return {'original_path': row['path'], 'resolved_path': str(target), **actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--map', action='append', default=[], metavar='OLD=NEW',
                        help='Explicit absolute prefix relocation; longest matching prefix wins')
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    root = args.project_root.absolute()
    mappings = []
    for item in args.map:
        old, separator, new = item.partition('=')
        if not separator or not Path(old).is_absolute() or not Path(new).is_absolute():
            parser.error('--map requires OLD=NEW with both prefixes absolute')
        if '..' in Path(old).parts or '..' in Path(new).parts or any(Path(old) == pair[0] for pair in mappings):
            parser.error('Duplicate or noncanonical mapping prefix')
        mappings.append((Path(old), Path(new)))
    manifest_hash = digest(args.manifest)
    data = json.loads(args.manifest.read_text())
    rows = list(records(data))
    if not rows:
        raise ValueError('No supported checksum identities in manifest')
    protected = {args.manifest.resolve()}
    for _, row in rows:
        try:
            protected.add(relocate(row.get('path'), root, mappings).resolve())
        except ValueError:
            pass  # Report the unmapped record below rather than skip its failure.
    if args.report.resolve() in protected:
        raise ValueError('Report would overwrite a manifest or verified input')
    passed, failures = [], []
    for pointer, row in rows:
        try:
            passed.append({'record': pointer, **verify_row(row, root, mappings)})
        except (ValueError, OSError) as error:
            failures.append({'record': pointer, 'error': str(error)})
    if digest(args.manifest) != manifest_hash:
        raise ValueError('Manifest changed during verification')
    report = {'scope': 'Only explicit path/bytes/sha256 identities and project-root files-map entries; not recursive manifest closure, scientific validation, or checkpoint resumability.',
              'manifest': {'path': str(args.manifest.absolute()), **manifest_hash},
              'verifier': {'path': str(Path(__file__).absolute()), **digest(Path(__file__))},
              'project_root': str(root), 'mappings': [[str(a), str(b)] for a, b in mappings],
              'records_checked': len(rows), 'verified_files': passed, 'failures': failures,
              'verified': not failures}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # A new report does not modify original identities, timestamps or run outputs.
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(f'{len(passed)}/{len(rows)} identities verified; {len(failures)} failures')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
