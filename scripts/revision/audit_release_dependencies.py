"""Inventory explicit file-identity references in a staged release's JSON files."""
import argparse
from collections import Counter
import json
from pathlib import Path

from stage_revision_release import digest


def identities(value, pointer='$'):
    if isinstance(value, dict):
        if value.get('schema') == 1 and value.get('path_base') == 'project root' and isinstance(value.get('files'), dict):
            for name, row in value['files'].items():
                yield pointer+'/files/'+name, dict(row, path=name)
            return
        if isinstance(value.get('path'), str) and ('bytes' in value or 'sha256' in value):
            yield pointer, value
        for key, child in value.items():
            yield from identities(child, pointer+'/'+str(key))
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from identities(child, pointer+'/'+str(i))


def classify(row, files, original_root):
    path = Path(row['path'])
    if '..' in path.parts:
        return 'invalid_path', None
    if path.is_absolute():
        if not path.is_relative_to(original_root):
            return 'external_absolute', None
        path = path.relative_to(original_root)
    relative = str(path)
    if relative not in files:
        return 'not_in_candidate', relative
    expected = files[relative]
    if 'sha256' not in row:
        return 'included_without_reference_hash', relative
    if row['sha256'] != expected['sha256'] or ('bytes' in row and row['bytes'] != expected['bytes']):
        return 'included_identity_mismatch', relative
    return 'included_identity_match', relative


def audit(candidate, original_root):
    candidate, original_root = Path(candidate).resolve(), Path(original_root).absolute()
    manifest_path = candidate/'release-candidate-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get('complete') or manifest.get('path_base') != 'project root':
        raise ValueError('Expected complete project-relative candidate manifest')
    files = manifest['files']
    unique, scanned, without = {}, 0, []
    for name, expected in files.items():
        if not name.endswith('.json'):
            continue
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe candidate member path')
        path = candidate/relative
        if digest(path) != {'bytes': expected['bytes'], 'sha256': expected['sha256']}:
            raise ValueError(f'Candidate JSON changed: {name}')
        rows = list(identities(json.loads(path.read_text())))
        scanned += 1
        if not rows:
            without.append(name)
        for pointer, row in rows:
            key = (row['path'], row.get('sha256'), row.get('bytes'))
            if key not in unique:
                status, member = classify(row, files, original_root)
                unique[key] = {'path': row['path'], 'sha256': row.get('sha256'),
                               'bytes': row.get('bytes'), 'status': status,
                               'candidate_member': member, 'references': []}
            unique[key]['references'].append({'json': name, 'pointer': pointer})
    records = sorted(unique.values(), key=lambda row: (row['status'], row['path'], str(row['sha256'])))
    return {'scope': 'Explicit path-plus-size/hash records and project-root file maps in candidate JSON only. Not dependency closure: arbitrary strings, file-relative names, runtime code dependencies and alternate checksum schemas are not resolved.',
            'candidate': str(candidate), 'manifest': digest(manifest_path),
            'original_root_mapping': str(original_root), 'json_files_scanned': scanned,
            'json_files_without_supported_identities': without,
            'unique_reference_variants': len(records),
            'status_counts': dict(Counter(row['status'] for row in records)),
            'references': records, 'dependency_closure_proven': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--original-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new dependency-report pathname')
    result = audit(args.candidate, args.original_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('json_files_scanned', 'unique_reference_variants', 'status_counts')}, indent=2))


if __name__ == '__main__':
    main()
