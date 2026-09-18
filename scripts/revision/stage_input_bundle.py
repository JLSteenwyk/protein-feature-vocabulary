"""Copy explicitly referenced project inputs into a resumable local-only bundle."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil

from stage_revision_release import digest


def input_plan(report):
    plan = {}
    for row in report['references']:
        if row['status'] != 'not_in_candidate':
            continue
        name = row['candidate_member']
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe input path')
        if path.parts[0] not in {'data', 'models', 'results'} and name != 'INTERPRETABILITY.zip':
            continue
        entry = plan.setdefault(name, {'recorded_hashes': [], 'recorded_sizes': []})
        if row['sha256'] is not None:
            entry['recorded_hashes'].append(row['sha256'])
        if row['bytes'] is not None:
            entry['recorded_sizes'].append(row['bytes'])
    for name, row in plan.items():
        row['recorded_hashes'] = sorted(set(row['recorded_hashes']))
        row['recorded_sizes'] = sorted(set(row['recorded_sizes']))
        if len(row['recorded_hashes']) > 1 or len(row['recorded_sizes']) > 1:
            raise ValueError(f'Conflicting historical input identities: {name}')
    return dict(sorted(plan.items()))


def publish(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dependency-report', required=True, type=Path)
    parser.add_argument('--project-root', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root, destination = args.project_root.resolve(), args.destination.absolute()
    if destination.is_relative_to(root) or root.is_relative_to(destination):
        raise ValueError('Bundle must be in a separate tree outside the source workspace')
    plan = input_plan(json.loads(args.dependency_report.read_text()))
    if not plan:
        raise ValueError('No eligible missing project inputs')
    run_identity = {'report': digest(args.dependency_report), 'source_root': str(root),
                    'plan': plan, 'script': digest(Path(__file__))}
    destination.mkdir(parents=True, exist_ok=args.resume)
    with (destination/'.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = destination/'input-bundle-manifest.json'
        if args.resume:
            manifest = json.loads(output.read_text())
            if manifest['run_identity'] != run_identity:
                raise ValueError('Resume inputs/code changed; use a new bundle')
        else:
            manifest = {'schema': 1, 'path_base': 'project root', 'algorithm': 'sha256',
                        'run_identity': run_identity, 'complete': False,
                        'publication_ready': False,
                        'scope': 'Local bundle of recognized missing project inputs, not full dependency closure or redistribution authorization. Fresh hashes of size-only references do not establish historical content identity.',
                        'files': {}}
            publish(output, manifest)
        for index, (name, expected) in enumerate(plan.items(), 1):
            source, target = root/name, destination/name
            current = digest(source)
            if expected['recorded_hashes'] and current['sha256'] != expected['recorded_hashes'][0]:
                raise ValueError(f'Historical input hash differs: {name}')
            if expected['recorded_sizes'] and current['bytes'] != expected['recorded_sizes'][0]:
                raise ValueError(f'Historical input size differs: {name}')
            if name in manifest['files']:
                saved = {k: manifest['files'][name][k] for k in ('bytes', 'sha256')}
                if saved != current or digest(target) != current:
                    raise ValueError(f'Completed bundle input changed: {name}')
                continue
            if target.exists():
                raise FileExistsError(f'Uncommitted destination: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.partial')
            shutil.copy2(source, temporary)
            if digest(temporary) != current or digest(source) != current:
                raise ValueError(f'Input changed or copy corrupt: {name}')
            temporary.replace(target)
            manifest['files'][name] = dict(current, historical_hash_verified=bool(expected['recorded_hashes']))
            publish(output, manifest)
            print(f'{index}/{len(plan)} verified {name}', flush=True)
        manifest.update(complete=True, expected_files=len(plan),
                        total_bytes=sum(row['bytes'] for row in manifest['files'].values()))
        publish(output, manifest)
        print(f'Complete: {len(plan)} inputs, {manifest["total_bytes"]} bytes', flush=True)


if __name__ == '__main__':
    main()
