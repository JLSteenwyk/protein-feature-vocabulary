"""Compare successful replay input identities with recorded staged inventories."""
from collections import Counter
import json
from pathlib import Path

from common import ROOT
from probe_checkpoint import atomic_json
from stage_revision_release import digest


REPORTS = ['relocated_go_check.json', 'relocated_head_summary_check.json',
           'relocated_interventions_separate_check.json', 'relocated_probe_check.json',
           'relocated_matching_check.json', 'relocated_matching_sensitivities_check.json',
           'relocated_cross_modal_check.json', 'relocated_reconstruction_check.json',
           'relocated_documents_complete_check.json']
PACKAGES = ['candidate_manifest.json', 'input_bundle_manifest.json',
            'input_addendum_manifest.json', 'public_code_manifest.json', 'external_model_manifest.json']


def group(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe replay member path')
    first = path.parts[0]
    if first in {'data', 'models', 'results'}:
        return 'scientific_inputs'
    if first == 'external_inputs':
        return 'external_model_inputs'
    if first in {'scripts', 'src', 'tests'}:
        return 'code'
    if first == 'revision':
        return 'revision_records_documents_assets'
    return 'other'


def matching_packages(name, expected, packages):
    matched, different = [], []
    for label, files in packages.items():
        if name not in files:
            continue
        if all(files[name].get(k) == expected[k] for k in ['bytes', 'sha256']):
            matched.append(label)
        else:
            different.append(label)
    return matched, different


def main():
    directory = ROOT/'revision/release'
    reports = {name: json.loads((directory/name).read_text()) for name in REPORTS}
    packages = {name: json.loads((directory/name).read_text()) for name in PACKAGES}
    if not all(value.get('complete') for value in [*reports.values(), *packages.values()]):
        raise ValueError('Only completed replay/staging records may enter this audit')
    inventories = {name: value['files'] for name, value in packages.items()}
    variants = {}
    for report_name, report in reports.items():
        for name, identity in report['copied_files'].items():
            key = (name, identity['bytes'], identity['sha256'])
            if key not in variants:
                variants[key] = {'path': name, 'group': group(name), 'bytes': identity['bytes'],
                                 'sha256': identity['sha256'], 'replays': []}
            variants[key]['replays'].append(report_name)
    counts, uncovered = {}, []
    for key in sorted(variants):
        row = variants[key]
        matched, different = matching_packages(row['path'], row, inventories)
        status = 'matched' if matched else ('different_content' if different else 'absent')
        counts.setdefault(row['group'], Counter())[status] += 1
        if not matched:
            uncovered.append({**row, 'status': status, 'different_packages': different})
    output = {'scope': 'Recorded copied-file identities from nine completed workflows compared with five local inventories; not fresh rehashing, dynamic dependency tracing, acquisition/licensing approval or whole-project dependency closure.',
              'inputs': {name: digest(directory/name) for name in REPORTS+PACKAGES},
              'script': digest(Path(__file__)), 'replays': REPORTS, 'packages': PACKAGES,
              'n_distinct_paths': len({key[0] for key in variants}),
              'n_identity_variants': len(variants),
              'groups': {name: dict(value) for name, value in counts.items()},
              'uncovered': uncovered,
              'all_observed_identities_covered': not uncovered,
              'explicitly_not_covered': ['Pending full probe-refit run',
                  'Unexecuted upstream extraction/training workflows',
                  'Native tools, installed packages, acquisition and distribution rights']}
    atomic_json(directory/'replay_bundle_coverage.json', output)
    print(json.dumps({key: output[key] for key in ['n_distinct_paths', 'n_identity_variants', 'groups',
                                                 'all_observed_identities_covered']}, indent=2))


if __name__ == '__main__':
    main()
