"""Build local, explicitly selected author-upload archives; never publish."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[2]
INPUT_PREFIXES = ('data/', 'results/scaled_1.5M/activations/',
                  'results/scaled_1.5M/cross_modal/',
                  'results/scaled_1.5M/sae_features/')
SAE_PREFIXES = ('models/sae/', 'models/sae_1.5M/')
SPLITS = ('revision/analyses/clusters.json',
          'revision/analyses/identity50_cluster.tsv',
          'revision/analyses/probe_split.json',
          'revision/analyses/probes_overlap_excluded/probe_split.json')


def identity(stream):
    h, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(block)
        size += len(block)
    return {'bytes': size, 'sha256': h.hexdigest()}


def safe_source(root, name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Unsafe path: {name}')
    source = root / name
    if source.is_symlink() or any(x.is_symlink() for x in source.parents):
        raise ValueError(f'Symlink: {name}')
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def input_kind(name):
    if name.startswith(SAE_PREFIXES) and name.endswith('/best.pt'):
        return 'sae'
    if name.startswith(INPUT_PREFIXES):
        return 'data'
    return None


def select(root):
    frozen = json.loads((root/'revision/release/input_bundle_manifest.json').read_text())
    if not frozen['complete']:
        raise ValueError('Incomplete input manifest')
    groups = {'code': {}, 'data': {}, 'sae': {}}
    for name, record in frozen['files'].items():
        kind = input_kind(name)
        if kind:
            groups[kind][name] = {k: record[k] for k in ('bytes', 'sha256')}
    for name in SPLITS:
        groups['data'][name] = None
    for tree in ('src', 'scripts', 'tests'):
        for path in sorted((root/tree).rglob('*.py')):
            relative = path.relative_to(root)
            if not any(p.startswith('.') or p == '__pycache__' for p in relative.parts):
                groups['code'][str(relative)] = None
    groups['code']['LICENSE'] = None
    for path in sorted((root/'revision/reproducibility').glob('requirements-*.txt')):
        groups['code'][str(path.relative_to(root))] = None
    for name, source in [('README.md', 'revision/release/author_upload/README.md'),
                         ('UPLOAD_INSTRUCTIONS.md', 'revision/release/author_upload/UPLOAD_INSTRUCTIONS.md')]:
        groups['code'][name] = {'source': source}
    return groups


def build(root, output, shard_bytes):
    root, output = root.resolve(), output.resolve()
    groups = select(root)
    output.mkdir(parents=True, exist_ok=False)
    report = {'complete': False, 'published': False, 'archives': {},
              'scope': 'Code, selected inputs and project SAEs; no corrected results, figures, private materials or third-party weights.'}
    report_path = output/'UPLOAD_MANIFEST.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    for kind, entries in groups.items():
        batches, batch, size = [], [], 0
        for name, expected in sorted(entries.items()):
            source = safe_source(root, (expected or {}).get('source', name))
            n = source.stat().st_size
            if batch and size+n > shard_bytes:
                batches.append(batch)
                batch, size = [], 0
            batch.append((name, source, expected))
            size += n
        if batch:
            batches.append(batch)
        for index, batch in enumerate(batches, 1):
            filename = f'protein-feature-vocabulary-{kind}-{index:02d}.tar'
            target = output/filename
            records = {}
            with tarfile.open(target, 'w') as archive:
                for name, source, expected in batch:
                    with source.open('rb') as stream:
                        before = identity(stream)
                    if expected and 'sha256' in expected and before != expected:
                        raise ValueError(f'Frozen identity mismatch: {name}')
                    info = tarfile.TarInfo(name)
                    info.size = before['bytes']
                    info.mode = 0o644
                    info.mtime = 0
                    with source.open('rb') as stream:
                        archive.addfile(info, stream)
                    records[name] = before
            with tarfile.open(target, 'r:') as archive:
                seen = set()
                for member in archive:
                    if not member.isfile() or member.name in seen or member.name not in records:
                        raise ValueError('Unexpected archive member')
                    seen.add(member.name)
                    with archive.extractfile(member) as stream:
                        if identity(stream) != records[member.name]:
                            raise ValueError(f'Archive byte mismatch: {member.name}')
                if seen != set(records):
                    raise ValueError('Missing archive member')
            with target.open('rb') as stream:
                record = identity(stream)
            record['files'] = records
            record['verified_readback'] = True
            report['archives'][filename] = record
            report_path.write_text(json.dumps(report, indent=2)+'\n')
            print(f'Verified {filename}: {len(records)} files, {record["bytes"]} bytes', flush=True)
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    checksum_lines = [f'{r["sha256"]}  {n}\n' for n, r in report['archives'].items()]
    with report_path.open('rb') as stream:
        checksum_lines.append(f'{identity(stream)["sha256"]}  UPLOAD_MANIFEST.json\n')
    (output/'SHA256SUMS').write_text(''.join(checksum_lines))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--shard-bytes', type=int, default=4_000_000_000)
    args = parser.parse_args()
    if args.shard_bytes <= 0:
        parser.error('--shard-bytes must be positive')
    build(args.root, args.output, args.shard_bytes)
