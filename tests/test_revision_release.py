"""Local staging must preserve paths/content and fail closed on unsafe inputs."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/revision'))
from stage_revision_release import TREES, selected, stage
from audit_release_dependencies import classify, identities
from stage_input_bundle import input_plan


def test_joined_package_rejects_conflicts_and_unsafe_paths(tmp_path):
    from assemble_revision_package import add
    first, second = tmp_path/'first', tmp_path/'second'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    plan = {}
    add(plan, 'data/item', first)
    add(plan, 'data/item', first)
    assert len(plan) == 1
    with pytest.raises(ValueError, match='Conflicting'):
        add(plan, 'data/item', second)
    for path in ['/outside', '../outside', 'revision-package-manifest.json']:
        with pytest.raises(ValueError, match='Unsafe'):
            add(plan, path, first)
    with pytest.raises(ValueError, match='Frozen input changed'):
        add(plan, 'data/other', first, {'bytes': 5, 'sha256': 'wrong'})
    link = tmp_path/'link'
    link.symlink_to(first)
    with pytest.raises(ValueError, match='Symlink'):
        add(plan, 'data/link', link)


def test_joined_package_includes_guidance_license_and_frozen_inputs(tmp_path, monkeypatch):
    import json
    import assemble_revision_package as module
    from stage_revision_release import digest
    from verify_relocated_manifest import records
    root, bundle, target = tmp_path/'project', tmp_path/'inputs', tmp_path/'joined'
    (root/'revision/release').mkdir(parents=True)
    bundle.mkdir()
    (root/'LICENSE').write_text('MIT fixture')
    (root/'code.py').write_text('pass')
    (root/'revision/release/guide.md').write_text('guide')
    (root/'revision/release/old.zip').write_bytes(b'excluded')
    (bundle/'input.txt').write_text('frozen')
    manifest = tmp_path/'inputs.json'
    manifest.write_text(json.dumps({'complete': True, 'path_base': 'project root',
                                   'files': {'input.txt': digest(bundle/'input.txt')}}))
    monkeypatch.setattr(module, 'candidates', lambda _: [root/'code.py'])
    report = module.assemble(root, target, [(manifest, bundle)])
    assert report['complete'] and not report['publication_ready']
    assert set(report['files']) == {'code.py', 'LICENSE', 'revision/release/guide.md', 'input.txt'}
    assert len(list(records(report))) == 4
    for name, expected in report['files'].items():
        assert digest(target/name) == expected
    with pytest.raises(FileExistsError):
        module.assemble(root, target, [(manifest, bundle)])


def test_editor_draft_collects_main_figures_only_and_rejects_paths(tmp_path):
    from stage_editor_draft import source_inputs
    journal, figures = tmp_path/'revision/journal', tmp_path/'revision/figures'
    journal.mkdir(parents=True)
    figures.mkdir()
    source = journal/'bioinformatics.tex'
    source.write_text('\n'.join(r'\includegraphics{f'+str(i)+'.png}' for i in range(4)))
    for i in range(4):
        (figures/f'f{i}.png').write_bytes(b'fixture')
    assert len(source_inputs(tmp_path)) == 10
    assert not any('supplement' in name or 'response' in name for name in source_inputs(tmp_path))
    source.write_text(source.read_text().replace('f0.png', '../f0.png'))
    with pytest.raises(ValueError, match='plain local filename'):
        source_inputs(tmp_path)


def test_replay_bundle_coverage_requires_content_match_not_only_path():
    from audit_replay_bundle_coverage import group, matching_packages
    expected = {'bytes': 4, 'sha256': 'expected'}
    packages = {'old': {'data/a': {'bytes': 4, 'sha256': 'old'}},
                'new': {'data/a': expected}}
    assert matching_packages('data/a', expected, packages) == (['new'], ['old'])
    assert matching_packages('data/missing', expected, packages) == ([], [])
    assert group('external_inputs/model.bin') == 'external_model_inputs'
    assert group('results/summary.h5') == 'scientific_inputs'
    for unsafe in ['/external', '../data/file']:
        with pytest.raises(ValueError, match='Unsafe'):
            group(unsafe)


def test_public_code_selection_excludes_correspondence_and_large_inputs(tmp_path):
    from stage_public_code import selection, stage, FIXTURES, DEPENDENCIES
    from verify_relocated_manifest import records
    names = list(FIXTURES)+['revision/reproducibility/'+p for p in DEPENDENCIES]
    names += ['LICENSE', 'revision/release/public_code/README.md', 'revision/release/public_code/.gitignore',
              'src/a.py', 'scripts/a.py', 'tests/test_a.py', 'scripts/.private/key.py',
              'revision/response_to_reviewers.md', 'revision/action_list.md',
              'data/input.h5', 'models/checkpoint.pt', 'revision/analyses/raw/p1.npz']
    for name in names:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    chosen = selection(tmp_path)
    assert chosen['README.md'] == tmp_path/'revision/release/public_code/README.md'
    assert set(chosen) == set(FIXTURES) | {'revision/reproducibility/'+p for p in DEPENDENCIES} | {
        'LICENSE', 'README.md', '.gitignore', 'src/a.py', 'scripts/a.py', 'tests/test_a.py'}
    manifest = stage(tmp_path, tmp_path.with_name(tmp_path.name+'-draft'))
    assert len(list(records(manifest))) == len(chosen)
    assert manifest['complete'] and not manifest['publication_ready']
    assert manifest['license'] == 'MIT'
    assert chosen['LICENSE'] == tmp_path/'LICENSE'
    (tmp_path/'src/link.py').symlink_to(tmp_path/'src/a.py')
    with pytest.raises(ValueError, match='Symlink'):
        selection(tmp_path)


def test_intervention_interval_formatting_preserves_signs_units_and_order():
    from audit_intervention_reported_numbers import interval_text
    assert interval_text({'ci95': [-.012345, .023456]}, scale=100) == '-1.23--2.35'
    assert interval_text({'bounds': [-.00007824, .00030256]}, key='bounds', precision=6,
                         separator='-') == '-0.000078-0.000303'
    with pytest.raises(ValueError, match='Reversed'):
        interval_text({'ci95': [1., -1.]})


def test_probe_replay_copy_resumes_only_matching_content(tmp_path):
    from check_relocated_probe_refits import checked_copy
    from stage_revision_release import digest
    source, target = tmp_path/'source', tmp_path/'copy/input'
    source.write_bytes(b'original')
    expected = digest(source)
    checked_copy(source, target, expected)
    checked_copy(source, target, expected)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Existing copy changed'):
        checked_copy(source, target, expected)
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='Source changed'):
        checked_copy(source, target, expected)


def test_refit_verifier_detects_prediction_change_with_unchanged_metrics(tmp_path, monkeypatch):
    import json
    import numpy as np
    import verify_portable_probe_refits as verifier
    from stage_revision_release import digest
    base, run = tmp_path/'base', tmp_path/'run'
    base.mkdir()
    run.mkdir()
    archived = base/'prediction.npz'
    current = run/'probe_predictions_composition_control.npz'
    values = {'y': np.array([0, 1]), 'score': np.array([0., 1.]),
              'protein': np.array(['a', 'b']), 'cluster': np.array(['a', 'b'])}
    np.savez(archived, **values)
    np.savez(current, **values)
    np.savez(run/'probe_fit_composition_control.npz', coef=np.array([[1.]]), intercept=np.array([0.]))
    output = {'sample_counts': {}, 'models': {'composition': {'control': {'auroc': 1.}}}}
    (run/'probe_controls.json').write_text(json.dumps(output))
    baseline = {**output, 'prediction_files': {'composition': {'control': {
        'path': str(archived.relative_to(tmp_path)), 'sha256': digest(archived)['sha256']}}}}
    (base/'combined_probe_controls.json').write_text(json.dumps(baseline))
    monkeypatch.setattr(verifier, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['verify', '--base-run-dir', str(base), '--run-dir', str(run),
                                     '--model', 'composition'])
    verifier.main()
    assert json.loads((run/'refit_verification.json').read_text())['complete']
    np.savez(current, **{**values, 'score': np.array([.1, 1.])})
    with pytest.raises(ValueError, match='prediction differs'):
        verifier.main()
    assert not json.loads((run/'refit_verification.json').read_text())['complete']


def test_portable_probe_split_rejects_leakage_bad_rows_and_exclusions():
    import copy
    import numpy as np
    from run_portable_probe_refits import validate_split
    ids = np.array(['a', 'b', 'c'])
    offsets = np.array([[0, 2], [2, 2], [4, 2]])
    rows = [np.array([0, 1]), np.array([2]), np.array([4, 5])]
    split = {'seed': 2288, 'train': ['a'], 'validation': ['b'], 'test': ['c']}
    clusters = {p: p for p in ids}
    assert validate_split(ids, offsets, rows, split, clusters)[2].tolist() == [2, 2]
    with pytest.raises(ValueError, match='cluster leakage'):
        validate_split(ids, offsets, rows, split, {**clusters, 'b': 'a'})
    with pytest.raises(ValueError, match='sampled residue'):
        validate_split(ids, offsets, [np.array([0, 0]), *rows[1:]], split, clusters)
    with pytest.raises(ValueError, match='outside'):
        validate_split(ids, offsets, [np.array([2]), *rows[1:]], split, clusters)
    excluded = copy.deepcopy(split)
    excluded['excluded_training_homologs'] = ['c']
    with pytest.raises(ValueError, match='excluded'):
        validate_split(ids, offsets, rows, excluded, clusters)


def test_cross_modal_replay_requires_ids_conditions_and_finite_means(tmp_path):
    import numpy as np
    from check_relocated_cross_modal import compare_means
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    values = {'ids': np.array(['a', 'b']), 's': np.array([[0., 1.], [2., 0.]]),
              'sst': np.array([[1., 0.], [0., 2.]])}
    np.savez(a, **values)
    np.savez(b, **values)
    assert compare_means(a, b)['sst']['exact']
    np.savez(b, **{**values, 'ids': values['ids'][::-1]})
    with pytest.raises(ValueError, match='array differs'):
        compare_means(a, b)
    np.savez(b, **{**values, 's': values['sst'], 'sst': values['s']})
    with pytest.raises(ValueError, match='array differs'):
        compare_means(a, b)
    np.savez(b, ids=values['ids'], s=values['s'])
    with pytest.raises(ValueError, match='inventory differs'):
        compare_means(a, b)
    values['s'][0, 0] = np.inf
    np.savez(a, **values)
    np.savez(b, **values)
    with pytest.raises(ValueError, match='Nonfinite'):
        compare_means(a, b)


def test_matching_sensitivity_comparison_preserves_undefined_positions(tmp_path):
    import numpy as np
    from check_relocated_matching_sensitivities import compare_arrays
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, ids=np.array(['a', 'b']), r=np.array([np.nan, .3]))
    np.savez(b, ids=np.array(['a', 'b']), r=np.array([np.nan, .3]))
    assert compare_arrays(a, b)['r']['exact']
    for values in [np.array([0., .3]), np.array([.3, np.nan]), np.array([np.nan, .4])]:
        np.savez(b, ids=np.array(['a', 'b']), r=values)
        with pytest.raises(ValueError, match='array differs'):
            compare_arrays(a, b)


def test_matching_relocation_requires_exact_pair_ids_values_and_dtypes(tmp_path):
    import numpy as np
    from check_relocated_matching import compare_pairs
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, ids=np.array([1, 2]), r=np.array([.3, .4], dtype=np.float32))
    np.savez(b, ids=np.array([1, 2]), r=np.array([.3, .4], dtype=np.float32))
    assert compare_pairs(a, b)['r']['exact']
    for ids, values in [(np.array([2, 1]), np.array([.3, .4], dtype=np.float32)),
                        (np.array([1, 2]), np.array([.3, .4], dtype=np.float64))]:
        np.savez(b, ids=ids, r=values)
        with pytest.raises(ValueError, match='array differs'):
            compare_pairs(a, b)
    np.savez(b, ids=np.array([1, 2]))
    with pytest.raises(ValueError, match='inventory differs'):
        compare_pairs(a, b)


def test_reconstruction_relocation_checks_arrays_and_only_ignores_records(tmp_path):
    import numpy as np
    from check_relocated_reconstruction import compare_arrays, scientific_summary
    source = {'models': {'esm3': {'r2': .75, 'records': {'mtime_ns': 1}}}, 'n': 2}
    assert scientific_summary(source) == {'models': {'esm3': {'r2': .75}}, 'n': 2}
    assert source['models']['esm3']['records'] == {'mtime_ns': 1}
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, rows=np.array([1, 2]), error=np.array([.1, .2]))
    np.savez(b, rows=np.array([1, 2]), error=np.array([.1, .2]))
    assert compare_arrays(a, b)['rows']['exact']
    np.savez(b, rows=np.array([2, 1]), error=np.array([.1, .2]))
    with pytest.raises(ValueError, match='array differs'):
        compare_arrays(a, b)
    np.savez(b, rows=np.array([1, 2]))
    with pytest.raises(ValueError, match='inventory differs'):
        compare_arrays(a, b)


def test_document_relocation_collects_analysis_directory_figure(tmp_path):
    from check_relocated_documents import figure_inputs
    manuscript = tmp_path/'revision/manuscript'
    manuscript.mkdir(parents=True)
    asset = tmp_path/'revision/analyses/probes/figure.png'
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'fixture')
    source = manuscript/'sn-article.tex'
    source.write_text(r'\includegraphics[width=\textwidth]{../analyses/probes/figure.png}')
    assert figure_inputs(tmp_path) == [asset.resolve()]
    asset.unlink()
    with pytest.raises(ValueError, match='Missing or external'):
        figure_inputs(tmp_path)


def test_core_number_audit_requires_context_and_numeric_boundaries():
    from audit_core_reported_numbers import paragraph, require
    assert paragraph('First\n\nTarget 78.03\\%\n\nLast', 'Target') == 'Target 78.03%'
    require('Target 78.03%', '78.03%')
    for invalid in ['178.03%', '-78.03%', '78.04%']:
        with pytest.raises(ValueError, match='does not match'):
            require(invalid, '78.03%')
    with pytest.raises(ValueError, match='does not match'):
        require('7,2670', '7,267')
    with pytest.raises(ValueError, match='Expected one paragraph'):
        paragraph('Target\n\nTarget', 'Target')


def test_historical_intervention_verifiers_retain_archived_hashes():
    import json
    import hashlib
    root = Path(__file__).resolve().parents[1]
    for run in ['circuit_controls/esm2', 'circuit_controls/esm3', 'steering_controls']:
        record = json.loads((root/'revision/analyses'/run/'verified_summary.json').read_text())['verifier']
        local = root/'scripts/revision'/Path(record['path']).name
        assert local.stat().st_size == record['bytes']
        assert hashlib.sha256(local.read_bytes()).hexdigest() == record['sha256']


def test_intervention_summary_comparison_limits_ignored_fields_and_tolerance():
    from check_relocated_interventions import compare_values, scientific_view
    old = {'identity': {'path': '/old'}, 'pairs': {'a': {'effects': {'path': '/old'},
                                                     'tested_edges': 5000, 'mean': 1.}}}
    expected = {'pairs': {'a': {'tested_edges': 5000, 'mean': 1.}}}
    assert scientific_view(old) == expected
    assert compare_values(expected, expected) == 0
    with pytest.raises(ValueError, match='numerical mismatch'):
        compare_values(1., 1.001)
    with pytest.raises(ValueError, match='field inventory'):
        compare_values(expected, {})
    with pytest.raises(ValueError, match='non-floating'):
        compare_values(5000, 5000.)


def test_in_memory_relocation_is_explicit_and_preserves_archive():
    from relocated_inputs import mapped_identity, parse_mappings
    source = {'inputs': [{'path': '/old/a', 'sha256': 'abc', 'bytes': 3}],
              'description': '/old/is-not-an-input'}
    maps = parse_mappings(['/old=/new'])
    output = mapped_identity(source, maps, Path('/new'))
    assert source['inputs'][0]['path'] == '/old/a'
    assert output['inputs'][0] == {'path': '/new/a', 'sha256': 'abc', 'bytes': 3}
    assert output['description'] == source['description']
    with pytest.raises(ValueError, match='explicit prefix'):
        mapped_identity(source, parse_mappings(['/different=/new']), Path('/new'))
    with pytest.raises(ValueError, match='Duplicate'):
        parse_mappings(['/old=/new', '/old=/another'])
    with pytest.raises(ValueError, match='absolute'):
        parse_mappings(['relative=/new'])


def test_relocated_head_comparison_preserves_scientific_fields_and_input_hashes():
    from check_relocated_head_summary import compare_summaries
    old = {'inputs': [{'path': '/old/a', 'sha256': 'abc', 'bytes': 3}],
           'conditions': {'a': {'mean': 1., 'ci95': [0., 2.]}}}
    new = dict(old, inputs=[dict(old['inputs'][0], path='/new/a')])
    compare_summaries(old, new, Path('/new'))
    with pytest.raises(ValueError, match='outside relocated'):
        compare_summaries(old, old, Path('/new'))
    with pytest.raises(ValueError, match='Scientific summaries'):
        compare_summaries(old, dict(new, conditions={}), Path('/new'))
    with pytest.raises(ValueError, match='contents differ'):
        compare_summaries(old, dict(new, inputs=[dict(new['inputs'][0], sha256='other')]), Path('/new'))


def test_input_bundle_limits_scope_and_rejects_conflicting_provenance():
    row = {'status': 'not_in_candidate', 'candidate_member': 'data/a', 'sha256': 'abc', 'bytes': 3}
    report = {'references': [row, dict(row, candidate_member='env/a'),
                             dict(row, status='external_absolute', candidate_member=None)]}
    assert list(input_plan(report)) == ['data/a']
    report['references'].append(dict(row, sha256='other'))
    with pytest.raises(ValueError, match='Conflicting historical'):
        input_plan(report)


def test_dependency_inventory_distinguishes_content_gaps():
    row = {'path': '/original/src/a.py', 'sha256': 'abc', 'bytes': 3}
    files = {'src/a.py': {'sha256': 'abc', 'bytes': 3}}
    assert classify(row, files, Path('/original')) == ('included_identity_match', 'src/a.py')
    assert classify(dict(row, sha256='other'), files, Path('/original'))[0] == 'included_identity_mismatch'
    assert classify({'path': row['path'], 'bytes': 3}, files, Path('/original'))[0] == 'included_without_reference_hash'
    assert classify(dict(row, path='/original-extra/a'), files, Path('/original'))[0] == 'external_absolute'
    assert classify(dict(row, path='data/a'), files, Path('/original'))[0] == 'not_in_candidate'
    assert classify(dict(row, path='../a'), files, Path('/original'))[0] == 'invalid_path'
    mapped = {'schema': 1, 'path_base': 'project root', 'files': files}
    assert list(identities(mapped))[0][1]['path'] == 'src/a.py'
    assert len(list(identities({'nested': [row, {'file': 'unsupported.npz'}]}))) == 1


def test_release_excludes_transient_and_hidden_files():
    for name in ['revision/analyses/cross_modal_cache/a.npz', 'src/__pycache__/a.pyc',
                 'revision/analyses/.writer.lock', 'revision/release/old.json',
                 'scripts/.env', 'revision/analyses/mmseqs_tmp/data']:
        assert not selected(Path(name))
    assert selected(Path('revision/analyses/steering_controls/records/p1.npz'))
    assert selected(Path('revision/journal/clean-engine.log'))


def test_release_preserves_content_and_refuses_reuse(tmp_path):
    source = tmp_path/'source'
    for name in TREES:
        (source/name).mkdir(parents=True)
    (source/'src/example.py').write_text('value = 3\n')
    (source/'action_list.txt').write_text('pending')
    (source/'requirements.txt').write_text('numpy')
    destination = tmp_path/'candidate'
    result = stage(source, destination)
    assert result['complete'] and not result['publication_ready']
    assert result['expected_files'] == 3
    assert (destination/'src/example.py').read_bytes() == (source/'src/example.py').read_bytes()
    with pytest.raises(FileExistsError):
        stage(source, destination)
    with pytest.raises(ValueError, match='separate tree'):
        stage(source, source/'candidate')
    (source/'src/external').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError, match='Symlink'):
        stage(source, tmp_path/'candidate2')
