"""Statistical invariants independent of protein models and external downloads."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/revision'))
import numpy as np
from scipy import sparse
from common import full_column_shuffle,within_protein_shuffle,standardize,labels,auc_bootstrap,metrics
from run_probe_controls import legacy_shuffle, retained_split_proteins
from probe_checkpoint import ProbeCheckpoint
from run_go_audit import bh
from secondary_structure import ss3_predictions, SS8_VOCAB
from paired_probe_comparisons import score_tables, weighted_metrics, compare
from sklearn.metrics import roc_auc_score, average_precision_score
import pytest
from recompute_cross_modal import classify, aligned_chunk_metadata
from export_go_tests import complete_term_test
from matching_sensitivity import breadth_best, union_support_r
from capture_reproducibility import stable_digest
from audit_attention_atlas import query_marginal_jsd
from run_aligned_esm2_probes import align_residue_rows
from combine_probe_runs import validate_selections, validate_prediction
import combine_probe_runs
import json
from build_cross_modal_detail import category_summary
from build_go_detail import validate_go_categories
from copy import deepcopy
from audit_contact_summary import eligible_pair_count, historical_estimated_length, load_historical_precision
from audit_lens_summary import validate_lens_records
from recompute_reconstruction import reconstruction_r2, read_sample
import h5py
from audit_autointerpretability import validation_indices, score_summary
from build_probe_detail import group_index, validate_control_set
from verify_exclusion_split import validate_membership
from audit_feature_biology import description_counts, join_weights
from audit_case_study_plot import support_summary
from check_intervention_runtime import differences
from run_corrected_head_evaluation import head_conditions, prediction_changes
from summarize_corrected_heads import cluster_mean_interval
from build_corrected_head_figure import validate_summary
from build_supplement import extract_extended_figures


def test_supplement_requires_complete_order_alt_text_and_internal_references():
    text = '\n'.join(r'\begin{figure}[ht]\label{ed'+str(i)+r'} Alt text: fixture.\end{figure}' for i in range(1, 12))
    assert len(extract_extended_figures(text)) == 11
    with pytest.raises(ValueError, match='ordered ED1'):
        extract_extended_figures(text.replace(r'\label{ed11}', r'\label{ed12}'))
    with pytest.raises(ValueError, match='alt text'):
        extract_extended_figures(text.replace('Alt text:', 'Description:', 1))
    with pytest.raises(ValueError, match='Unresolved standalone'):
        extract_extended_figures(text.replace('fixture.', r'\ref{fig1}', 1))


def test_head_figure_rejects_unverified_or_incomplete_cohorts():
    with pytest.raises(ValueError, match='complete verified'):
        validate_summary({'raw_logits_verified': False})
    with pytest.raises(ValueError, match='complete verified'):
        validate_summary({'raw_logits_verified': True, 'successful': 99, 'failures': []})


def test_head_uncertainty_resamples_clusters_but_targets_protein_mean():
    result = cluster_mean_interval([1, 1, 0], ['a', 'a', 'b'], n=100)
    assert result['mean'] == 2/3
    assert result['n_clusters'] == 2
    assert result['n_proteins'] == 3
    assert result['ci95'] == [0, 1]
    assert cluster_mean_interval([0, 0], ['a', 'b'], n=10)['ci95'] == [0, 0]


def test_head_conditions_are_seeded_unique_and_keep_controls_separate():
    conditions = head_conditions()
    assert conditions == head_conditions()
    assert len(conditions) == 14
    heads = [h for _, h in conditions if h is not None]
    assert len(heads) == len(set(heads)) == 12
    assert conditions[2] == ('target_L0H7', (0, 7))


def test_corrected_prediction_changes_ignore_special_ss_tokens():
    sequence = np.array([[2., 0.], [0., 2.]])
    ss = np.zeros((2, 11))
    ss[:, 4] = 8
    other = ss.copy()
    other[:, 0] = 100
    result = prediction_changes(sequence, sequence, ss, other)
    assert result['sequence_disagreement'] == 0
    assert result['ss3_disagreement'] == 0
    assert result['mean_sequence_kl_normal_to_perturbed'] == 0
    other[:, 7] = 12
    assert prediction_changes(sequence, sequence, ss, other)['ss3_disagreement'] == 1
    with pytest.raises(ValueError, match='Unpaired'):
        prediction_changes(sequence, sequence[:1], ss, other)


def test_runtime_comparison_reports_error_and_nonfinite_values():
    import torch
    x = torch.tensor([1., 2.])
    assert differences(x, x) == {'max_abs': 0., 'relative_l2': 0., 'finite': True}
    assert differences(x, x+1)['max_abs'] == 1
    assert not differences(x, torch.tensor([float('nan'), 2.]))['finite']


def test_case_summary_identifies_truncated_support_without_reconstructing_values():
    result = support_summary([{'active_positions': [1, 3], 'n_active_residues': 9},
                              {'active_positions': [0], 'n_active_residues': 1}])
    assert result == {'features': 2, 'truncated_features': 1, 'saved_positions': 3,
                      'reported_active_positions': 10}
    with pytest.raises(ValueError, match='Inconsistent'):
        support_summary([{'active_positions': [1, 1], 'n_active_residues': 2}])


def test_description_unknown_text_is_not_semantic_classification():
    records = [{'motif_type': 'unknown', 'description': '```json incomplete'},
               {'motif_type': 'unknown', 'description': 'other'},
               {'motif_type': 'sequence_pattern', 'description': 'unvalidated'}]
    counts, missing = description_counts(records)
    assert counts == {'unknown': 2, 'sequence_pattern': 1}
    assert missing == {'unknown': 2, 'unknown_json_like': 1, 'unknown_other': 1}


def test_historical_weights_join_by_id_not_position():
    weights = [{'feature_id': 7, 'weight': -.5, 'abs_weight': .5}]
    categories = [{'feature_id': 3, 'category': 'enhanced'},
                  {'feature_id': 7, 'category': 'unclassified'}]
    assert join_weights(weights, categories)[0]['category'] == 'unclassified'
    with pytest.raises(ValueError, match='Duplicate category'):
        join_weights(weights, categories+categories)
    with pytest.raises(ValueError, match='Invalid probe/category'):
        join_weights(weights, categories[:1])


def test_exclusion_verifier_retains_partitions_and_rejects_cluster_overlap():
    primary = {'train': ['a', 'b'], 'validation': ['c'], 'test': ['d']}
    retained = {'train': ['a'], 'validation': ['c'], 'test': ['d'],
                'excluded_training_homologs': ['b']}
    clusters = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    assert validate_membership(primary, retained, {'b'}, clusters) == {
        'train': 1, 'validation': 1, 'test': 1}
    with pytest.raises(ValueError, match='partition mismatch'):
        validate_membership(primary, primary, {'b'}, clusters)
    clusters['d'] = 1
    with pytest.raises(ValueError, match='crosses partitions'):
        validate_membership(primary, retained, {'b'}, clusters)


def test_probe_control_set_checks_names_not_only_count():
    names = ['intact', 'binary_support', 'protein_mean_only']
    names += [f'{prefix}_{seed}' for prefix in
              ['legacy_support_preserving', 'global_column_shuffle', 'within_protein_shuffle']
              for seed in [2288, 2289, 2290]]
    report = {'completed_controls': 26, 'models': {
        'esm3': dict.fromkeys(names), 'esm2': dict.fromkeys(names),
        'composition': dict.fromkeys(['protein_composition_length', 'residue_identity_plus_composition'])}}
    validate_control_set(report)
    report['models']['esm2']['intact_typo'] = report['models']['esm2'].pop('intact')
    with pytest.raises(ValueError, match='unexpected controls'):
        validate_control_set(report)


def test_probe_figure_preserves_control_and_seed_identity():
    assert group_index('intact') == 0
    assert group_index('protein_mean_only') == 5
    assert group_index('global_column_shuffle_2288') == 3
    assert group_index('global_column_shuffle_2290') == 3
    with pytest.raises(ValueError, match='Unknown probe'):
        group_index('global_column_shuffle_9999')


def test_metadata_validation_selection_is_ordered_unique_and_held_out():
    values = np.arange(100, dtype=float)
    selected = validation_indices(values, np.random.RandomState(42), start=40)
    assert 23 <= len(selected) <= 30
    assert len(set(selected)) == len(selected)
    assert np.all(selected >= 40)
    assert selected[:8].tolist() == list(range(99, 91, -1))
    assert np.array_equal(selected, validation_indices(values, np.random.RandomState(42), start=40))


def test_metadata_score_missingness_is_not_parse_success():
    records = [{'score': .7, 'actual_values': [0, 1]},
               {'score': None, 'actual_values': [0, 1]},
               {'score': float('nan'), 'actual_values': [0, 0]},
               {'score': float('nan'), 'actual_values': [0, 1]}]
    summary = score_summary(records, 'score')
    assert summary['n_finite'] == 1
    assert summary['n_null'] == 1
    assert summary['n_nonfinite_nonnull'] == 2
    assert summary['n_nonfinite_with_constant_actual'] == 1
    assert summary['median'] == .7
    with pytest.raises(ValueError, match='constant actual'):
        score_summary([{'score': .7, 'actual_values': [0, 0]}], 'score')


def test_centered_reconstruction_r2_penalizes_constant_error():
    x = np.array([[1., 2.], [3., 4.]])
    prediction = x+10
    # Variance of a constant per-vector error is zero, but reconstruction is poor.
    assert np.allclose(1-(x-prediction).var(axis=1)/x.var(axis=1), 1)
    r2 = reconstruction_r2(np.square(x-prediction).sum(), x.sum(0), np.square(x).sum(), len(x))
    assert r2 == -99
    assert reconstruction_r2(0, x.sum(0), np.square(x).sum(), len(x)) == 1


def test_reconstruction_sample_joins_protein_positions(tmp_path):
    path = tmp_path/'sample.h5'
    with h5py.File(path, 'w') as f:
        f['ids'] = np.array([b'b', b'a'])
        f['offsets'] = [[0, 0, 2], [1, 2, 2]]
        f['activations'] = [[20, 21], [22, 23], [10, 11], [12, 13]]
    x = read_sample([path], np.array(['a', 'b']), np.array([2, 2]), np.array([0, 1]), np.array([1, 0]), 2)
    assert x.tolist() == [[12, 13], [20, 21]]
    with pytest.raises(ValueError, match='length mismatch'):
        read_sample([path], np.array(['a', 'b']), np.array([3, 2]), np.array([0, 1]), np.array([1, 0]), 2)


def test_lens_audit_rejects_inconsistent_condition_summaries():
    data = {'layers': [0], 'per_layer': {'0': {
        'accuracy_S': .2, 'accuracy_SSt': .3, 'accuracy_diff': .1,
        'kl_S': 2, 'kl_SSt': 3, 'kl_diff': 1}}}
    result = validate_lens_records(data)
    assert result['SSt_higher_agreement_blocks'] == [0]
    assert result['SSt_lower_KL_blocks'] == []
    wrong = deepcopy(data)
    wrong['per_layer']['0']['kl_diff'] = -1
    with pytest.raises(ValueError, match='difference inconsistent'):
        validate_lens_records(wrong)
    wrong = deepcopy(data)
    wrong['layers'] = [0, 2]
    with pytest.raises(ValueError, match='Layer records'):
        validate_lens_records(wrong)


def test_contact_supervised_length_bug_is_five_residues():
    for length in [80, 100, 200, 300]:
        pairs = sum(length-i-6 for i in range(length-6))
        assert eligible_pair_count(length) == pairs
        assert historical_estimated_length(pairs) == length-5


def test_historical_contact_L_is_prediction_budget_not_separation():
    precision = load_historical_precision()
    length = 8
    prediction = np.zeros((length, length))
    truth = np.zeros((length, length), dtype=bool)
    for i, j, score, contact in [(0, 6, 1, True), (0, 7, 3, False), (1, 7, 1, True)]:
        prediction[i, j] = prediction[j, i] = score
        truth[i, j] = truth[j, i] = contact
    result = precision(prediction, truth, length)
    assert result['L/5'] == 0
    assert result['L/2'] == pytest.approx(2/3)
    assert result['L'] == pytest.approx(2/3)


def test_go_figure_checks_denominators_and_matched_ids():
    model = {'n_active': 4, 'per_feature': [
        {'feature_id': 0, 'category': 'enhanced', 'n_terms_all_background_bh': 0, 'n_study_proteins': 1},
        {'feature_id': 1, 'category': 'enhanced', 'n_terms_all_background_bh': 4, 'n_study_proteins': 10},
        {'feature_id': 2, 'category': 'invariant', 'n_terms_all_background_bh': 2, 'n_study_proteins': 10},
        {'feature_id': 3, 'category': 'invariant', 'n_terms_all_background_bh': 6, 'n_study_proteins': 20}],
        'categories': {
            'enhanced': {'n_features': 2, 'n_enriched': 1, 'frac_enriched': .5,
                         'median_all_features': 2, 'median_enriched_only': 4},
            'invariant': {'n_features': 2, 'n_enriched': 2, 'frac_enriched': 1,
                          'median_all_features': 4, 'median_enriched_only': 4}}}
    balance = {'n_pairs': 1, 'outcome': {'median_enhanced': 4, 'median_unclassified': 2},
               'pairs': [{'enhanced_feature_id': 1, 'unclassified_feature_id': 2,
                          'enhanced_study_size': 10, 'unclassified_study_size': 10}]}
    assert validate_go_categories(model, balance) == [[4], [2]]
    wrong = deepcopy(model)
    wrong['categories']['enhanced']['median_enriched_only'] = 2
    with pytest.raises(ValueError, match='Enriched-only median'):
        validate_go_categories(wrong, balance)
    wrong = deepcopy(balance)
    wrong['pairs'][0]['unclassified_feature_id'] = 1
    with pytest.raises(ValueError, match='category or study size'):
        validate_go_categories(model, wrong)
    wrong = deepcopy(balance)
    wrong['pairs'] *= 2
    with pytest.raises(ValueError, match='reuses'):
        validate_go_categories(model, wrong)


def test_cross_modal_figure_uses_complete_categories_and_fixed_ranks():
    records = [{'feature_id': c*11+i, 'category': name,
                'paired_standardized_difference_ddof0': float(i)}
               for c, name in enumerate(['enhanced', 'unclassified', 'suppressed'])
               for i in range(11)]
    summary = category_summary(records[::-1])
    for c, name in enumerate(['enhanced', 'unclassified', 'suppressed']):
        assert summary[name]['n'] == 11
        assert summary[name]['median_effect'] == 5
        assert [r['feature_id'] for r in summary[name]['illustrated_records']] == [c*11+i for i in [1, 3, 5, 7, 9]]


def test_combined_snapshot_preserves_sources_and_composition_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(combine_probe_runs, 'ROOT', tmp_path)
    base, extra = tmp_path/'base', tmp_path/'extra'
    prediction = {'y': [0, 1], 'protein': ['a', 'b'], 'cluster': ['x', 'y'], 'score': [.1, .9]}
    for directory, model in [(base, 'esm3'), (extra, 'esm2')]:
        directory.mkdir()
        (directory/'probe_split.json').write_text('{"train": [], "validation": [], "test": ["a", "b"]}')
        np.savez(directory/'probe_rows.npz', train=[], validation=[], test=[0, 1])
        records = {'seed': 2288, 'sample_counts': {'test': 2}, 'models': {model: {'intact': {'auroc': 1}}}}
        if model == 'esm2':
            records['models']['composition'] = {'control': {'auroc': .5}}
        (directory/'probe_controls.json').write_text(json.dumps(records))
        np.savez(directory/f'probe_predictions_{model}_intact.npz', **prediction)
    (base/'composition_controls.json').write_text('{"control": {"auroc": 1}}')
    np.savez(base/'probe_predictions_composition_control.npz', **prediction)
    before = (base/'probe_controls.json').read_bytes()
    result = combine_probe_runs.combine(base, extra)
    assert result['completed_controls'] == 3
    assert result['models']['composition']['control']['auroc'] == 1
    assert result['prediction_files']['esm2']['intact']['path'].startswith('extra/')
    assert (base/'probe_controls.json').read_bytes() == before
    np.savez(extra/'probe_predictions_esm2_intact.npz', **{**prediction, 'protein': ['b', 'a']})
    with pytest.raises(ValueError, match='alignment'):
        combine_probe_runs.combine(base, extra)


def test_combined_probes_reject_misaligned_predictions():
    reference = {'y': np.array([0, 1]), 'protein': np.array(['a', 'b']),
                 'cluster': np.array(['x', 'y']), 'score': np.array([.2, .8])}
    validate_prediction(reference, reference)
    for key in ['y', 'protein', 'cluster']:
        wrong = {**reference, key: reference[key][::-1]}
        with pytest.raises(ValueError, match='alignment'):
            validate_prediction(reference, wrong)
    with pytest.raises(ValueError, match='scores'):
        validate_prediction(reference, {**reference, 'score': np.array([np.nan, 0])})


def test_combined_probes_reject_different_residue_selections(tmp_path):
    a, b = tmp_path/'a', tmp_path/'b'
    for directory in [a, b]:
        directory.mkdir()
        (directory/'probe_split.json').write_text('{"train": ["a"], "validation": ["b"], "test": ["c"]}')
        np.savez(directory/'probe_rows.npz', train=[0, 1], validation=[2], test=[3])
    validate_selections(a, b)
    np.savez(b/'probe_rows.npz', train=[1, 0], validation=[2], test=[3])
    with pytest.raises(ValueError, match='Residue selection'):
        validate_selections(a, b)


def test_shuffle_moves_support_and_preserves_column_distribution():
    x=sparse.csr_matrix(np.vstack([np.eye(8),np.zeros((24,8))]))
    z=full_column_shuffle(x,2288)
    assert np.array_equal(x.sum(0),z.sum(0))
    assert (x!=z).nnz>0
    assert np.array_equal(np.sort(x.toarray(),axis=0),np.sort(z.toarray(),axis=0))


def test_within_protein_shuffle_preserves_protein_means():
    x=sparse.csr_matrix(np.arange(48).reshape(12,4))
    groups=np.repeat([0,1,2],4)
    z=within_protein_shuffle(x,groups,2288)
    for g in range(3):assert np.array_equal(x[groups==g].sum(0),z[groups==g].sum(0))


def test_correlation_matches_numpy_and_flags_constants():
    x=np.random.default_rng(4).normal(size=(50,3));x[:,2]=1
    z,valid=standardize(x)
    assert np.allclose((z.T@z)[:2,:2],np.corrcoef(x[:,:2],rowvar=False),atol=1e-6)
    assert valid.tolist()==[True,True,False]


def test_bh_includes_absent_terms():
    assert np.allclose(bh(np.array([.001,.02]),10),[.01,.1])


def test_labels_use_one_based_inclusive_coordinates():
    meta={'p':{'features':[{'type':'Active site','start':2,'end':2},{'type':'Disulfide bond','start':1,'end':5}]}}
    assert labels(['p'],np.array([[0,5]]),meta).tolist()==[0,1,0,0,0]
    assert labels(['p'],np.array([[0,5]]),meta,broad=True).tolist()==[1,1,0,0,1]


def test_cluster_bootstrap_respects_perfect_and_tied_scores():
    y=np.tile([0,1],10);g=np.repeat(np.arange(10),2)
    assert auc_bootstrap(y,y,g,n=30)['low']==1
    assert auc_bootstrap(y,np.ones(20),g,n=30)['high']==.5


def test_legacy_shuffle_preserves_support():
    x = sparse.csr_matrix([[1., 0.], [2., 0.], [3., 8.], [0., 0.], [0., 9.]])
    shuffled = legacy_shuffle(x, 2288)
    assert np.array_equal(x.toarray() != 0, shuffled.toarray() != 0)
    assert np.array_equal(np.sort(x.toarray(), axis=0), np.sort(shuffled.toarray(), axis=0))


def test_protein_constant_predictions_cannot_localize():
    y = np.array([0, 1, 1, 0, 1, 0])
    group = np.repeat([0, 1], 3)
    result = metrics(y, np.repeat([.9, .1], 3), group)
    assert result['macro_within_protein_auroc'] == .5
    assert result['n_proteins_with_both_classes'] == 2


def test_ss3_excludes_special_tokens_and_maps_all_biological_states():
    logits = np.full((8, 11), -20.)
    logits[:, :3] = 10000.
    logits[np.arange(8), np.arange(3, 11)] = 20.
    assert ss3_predictions(logits).tolist() == [0, 0, 0, 2, 1, 1, 2, 2]
    order = np.arange(11)[::-1]
    assert np.array_equal(ss3_predictions(logits),
                          ss3_predictions(logits[:, order], tuple(SS8_VOCAB[i] for i in order)))


def test_ss3_uses_grouped_probability_and_validates_dimensions():
    logits = np.full((1, 11), -20.)
    logits[0, [3, 4, 5]] = 0.
    logits[0, 7] = .5
    assert ss3_predictions(logits).item() == 0
    with pytest.raises(ValueError):
        ss3_predictions(np.zeros((2, 3)))


def test_weighted_score_tables_equal_sklearn_with_ties():
    y = np.array([0, 1, 1, 0, 1, 0])
    score = np.array([.1, .4, .4, .4, .9, .8])
    groups = np.repeat(np.arange(3), 2)
    weights = np.array([2, 0, 3])
    actual = weighted_metrics(score_tables(y, score, groups, 3), weights)
    expected = [roc_auc_score(y, score, sample_weight=weights[groups]),
                average_precision_score(y, score, sample_weight=weights[groups])]
    assert np.allclose(actual, expected)


def test_paired_identical_predictions_have_zero_difference():
    data = {'y': np.tile([0, 1], 5), 'score': np.tile([.1, .7], 5),
            'protein': np.repeat(np.arange(5), 2), 'cluster': np.repeat(np.arange(5), 2)}
    result = compare(data, data, n_resamples=20)
    for metric in result['metrics'].values():
        assert metric['difference'] == 0
        assert metric['ci95'] == [0, 0]


def test_cross_modal_classification_retains_ineligible_family_members():
    delta = np.zeros((20, 4))
    delta[:, 0] = np.linspace(.5, 1.5, 20)
    delta[:, 1] = -delta[:, 0]
    delta[0, 2] = 1.
    _, _, _, p, q, category, eligible = classify(delta, np.array([True, True, True, False]))
    assert category.tolist() == ['enhanced', 'suppressed', 'unclassified', 'inactive']
    assert eligible.tolist() == [True, True, False, False]
    assert p[2] == q[2] == 1


def test_cross_modal_alignment_rejects_mismatched_proteins():
    a = {'ids': np.array([b'a', b'b']), 'offsets': np.array([[0, 0, 2], [1, 2, 3]]),
         'activations': np.zeros((5, 4))}
    b = {**a, 'ids': np.array([b'b', b'a'])}
    with pytest.raises(ValueError, match='IDs/order'):
        aligned_chunk_metadata(a, b)
    ids, offsets = aligned_chunk_metadata(a, a)
    assert ids.tolist() == ['a', 'b']
    assert offsets.shape == (2, 3)


def test_probe_exclusions_preserve_original_cluster_partitions():
    groups = np.array(['a', 'a', 'b', 'c', 'c', 'd'])
    ids = np.array(['p0', 'p1', 'p2', 'p3', 'p4', 'p5'])
    splits = [np.array(['a', 'b']), np.array(['c']), np.array(['d'])]
    retained = retained_split_proteins(groups, splits, ids, ['p1', 'p3'])
    assert [part.tolist() for part in retained] == [[0, 2], [4], [5]]
    assert not set(ids[np.concatenate(retained)]) & {'p1', 'p3'}


def test_probe_checkpoint_reuses_only_aligned_complete_predictions(tmp_path):
    checkpoint = ProbeCheckpoint(tmp_path, {'protocol': 1})
    expected = {'y': np.array([0, 1]), 'protein': np.array(['a', 'b']),
                'cluster': np.array(['x', 'y'])}
    output = {'models': {'model': {'intact': {'auroc': 1.}}}}
    assert not checkpoint.reusable(output, 'model', 'intact', expected)
    checkpoint.commit(output, 'model', 'intact', {**expected, 'score': np.array([.1, .9])},
                      {'coef': np.array([[1.]]), 'intercept': np.array([0.]), 'classes': np.array([0, 1])})
    assert checkpoint.reusable(output, 'model', 'intact', expected)
    with pytest.raises(ValueError, match='alignment'):
        checkpoint.reusable(output, 'model', 'intact', {**expected, 'y': np.array([1, 0])})
    checkpoint.close()
    resumed = ProbeCheckpoint(tmp_path, {'protocol': 1})
    assert resumed.load({}) == output
    resumed.close()
    with pytest.raises(ValueError, match='changed'):
        ProbeCheckpoint(tmp_path, {'protocol': 2})


def test_probe_checkpoint_rejects_duplicate_writer_and_legacy_results(tmp_path):
    checkpoint = ProbeCheckpoint(tmp_path, {'protocol': 1})
    with pytest.raises(BlockingIOError):
        ProbeCheckpoint(tmp_path, {'protocol': 1})
    checkpoint.close()
    legacy = tmp_path / 'legacy'
    legacy.mkdir()
    (legacy / 'probe_controls.json').write_text('{}')
    with pytest.raises(ValueError, match='Legacy'):
        ProbeCheckpoint(legacy, {'protocol': 1})


def test_complete_go_family_includes_absent_terms_and_ineligible_features():
    overlap = np.array([5, 0, 1, 0])
    population = np.array([5, 15, 30, 10])
    p, q, enriched = complete_term_test(overlap, population, 100, 5, True)
    assert p[1] == p[3] == q[1] == q[3] == 1
    assert np.isclose(q[0], p[0] * 4)
    assert enriched.tolist() == [True, False, False, False]
    p, q, enriched = complete_term_test(overlap, population, 100, 5, False)
    assert np.all(p == 1) and np.all(q == 1) and not enriched.any()


def test_breadth_matching_excludes_rare_and_incomparable_candidates():
    corr = np.array([[.99, .7, .2], [.3, .8, .99], [.9, .8, .7]])
    best, counts = breadth_best(corr, [10, 100, 3], [3, 20, 100])
    assert best.tolist() == [1, 2, -1]
    assert counts.tolist() == [1, 1, 0]


def test_union_support_correlation_omits_joint_zeros_only():
    a = np.array([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [0., 0.]])
    b = np.array([[0., 0.], [0., 0.], [2., 0.], [1., 2.], [0., 0.]])
    actual, count = union_support_r(a, b, [0, 1], [0, 1])
    assert np.isclose(actual[0], np.corrcoef(a[1:4, 0], b[1:4, 0])[0, 1])
    assert np.isnan(actual[1])
    assert count.tolist() == [3, 1]


def test_supplement_render_checks_order_references_and_bounds():
    from verify_supplement import check_rendered_pages

    def page(text):
        words = ''.join(f'<word xMin="1" xMax="2" yMin="1" yMax="2">{word}</word>'
                        for word in text.split())
        return f'<page width="100" height="100">{words}</page>'

    xml = '<html xmlns="http://www.w3.org/1999/xhtml">' + page('Title')
    xml += ''.join(page(f'Extended Data Fig. {i}: Caption Alt text: Description')
                   for i in range(1, 12)) + '</html>'
    assert len(check_rendered_pages(xml)) == 12
    expanded = xml.replace('</html>', page('Supplementary Methods Content') + page('References Entries') + '</html>')
    assert len(check_rendered_pages(expanded, expect_methods=True)) == 14
    with pytest.raises(ValueError, match='Methods or References'):
        check_rendered_pages(expanded.replace('References', 'Absent'), expect_methods=True)
    with pytest.raises(ValueError, match='bounds'):
        check_rendered_pages(xml.replace('xMax="2"', 'xMax="101"', 1))
    with pytest.raises(ValueError, match='Unresolved'):
        check_rendered_pages(xml.replace('Title', '??'))
    with pytest.raises(ValueError, match='misordered'):
        check_rendered_pages(xml.replace('>11:<', '>10:<'))
    with pytest.raises(ValueError, match='eleven'):
        check_rendered_pages('<html/>' )


def test_residual_preserving_intervention_and_matched_comparator():
    import torch
    from feature_interventions import intervene_hidden

    class BiasedSAE:
        def encode(self, x):
            return x.clone()

        def decode(self, z):
            return z * .5 + 7

    hidden = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
    sae = BiasedSAE()
    baseline = intervene_hidden(hidden, sae, [], 'reconstruction')
    removed = intervene_hidden(hidden, sae, [0], 'reconstruction')
    residual = intervene_hidden(hidden, sae, [0])
    assert torch.equal(intervene_hidden(hidden, sae, []), hidden)
    assert not torch.equal(baseline, hidden)
    assert torch.equal(residual-hidden, removed-baseline)
    assert torch.equal(residual[..., 0], hidden[..., 0]*.5)
    assert torch.equal(residual[..., 1], hidden[..., 1])
    mask = torch.tensor([[False, True, False]])
    masked = intervene_hidden(hidden, sae, [0], token_mask=mask)
    assert torch.equal(masked[:, [0, 2]], hidden[:, [0, 2]])
    assert torch.equal(masked[:, 1], residual[:, 1])
    assert torch.equal(hidden, torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]]))
    for bad in [[-1], [2], [0, 0], [True], [0.5]]:
        with pytest.raises(ValueError):
            intervene_hidden(hidden, sae, bad)
    with pytest.raises(ValueError, match='Token mask'):
        intervene_hidden(hidden, sae, [0], token_mask=mask.float())


def test_feature_hook_preserves_tuple_metadata_and_removes_on_failure():
    import torch
    from feature_interventions import feature_intervention

    class Block(torch.nn.Module):
        def forward(self, x):
            return x, 'metadata'

    class SAE:
        def encode(self, x):
            return x.clone()

        def decode(self, z):
            return z*.5

    block = Block()
    hidden = torch.ones(1, 2, 3)
    with pytest.raises(RuntimeError, match='deliberate'):
        with feature_intervention(block, SAE(), [0]):
            actual, metadata = block(hidden)
            assert metadata == 'metadata'
            assert torch.equal(actual[..., 0], hidden[..., 0]*.5)
            raise RuntimeError('deliberate')
    assert torch.equal(block(hidden)[0], hidden)
    assert len(block._forward_hooks) == 0


def test_circuit_cohort_selection_is_cluster_disjoint_and_reproducible():
    from run_circuit_controls import select_cohort, rank_features, conditions
    sequences = {str(i): 'A'*60 for i in range(12)}
    metadata = {p: {'taxid': '9606'} for p in sequences}
    clusters = {p: str(int(p)//2) for p in sequences}
    actual = select_cohort(sequences, metadata, clusters, 2, 3)
    assert actual == select_cohort(sequences, metadata, clusters, 2, 3)
    selected = actual['discovery'] + actual['evaluation']
    assert len({clusters[p] for p in selected}) == 5
    assert len(selected) == len(set(selected))
    assert rank_features([.2, .5, .5, 0], 2) == [1, 2]
    with pytest.raises(ValueError, match='active'):
        rank_features([.2, 0], 2)
    with pytest.raises(ValueError, match='Insufficient'):
        select_cohort(sequences, metadata, clusters, 4, 3)
    rows = conditions([1, 2])
    assert len(rows) == 7
    assert rows[:3] == [('normal', None, []), ('noop', 'residual_preserving', []),
                        ('reconstruction', 'reconstruction', [])]
    assert rows[3][2] == rows[4][2] == [1]


def test_circuit_commits_detect_corruption_and_uncommitted_arrays(tmp_path):
    from run_circuit_controls import committed, save_arrays
    path = tmp_path/'protein.npz'
    assert committed(path) is None
    save_arrays(path, {'mean': np.array([1., 2.])}, {'protein': 'test'})
    assert committed(path)['protein'] == 'test'
    path.write_bytes(b'corruption')
    with pytest.raises(ValueError, match='Corrupt'):
        committed(path)
    with pytest.raises(ValueError, match='Nonfinite'):
        save_arrays(tmp_path/'other.npz', {'mean': np.array([np.nan])}, {})


def test_circuit_paired_effects_separate_reconstruction_from_removal():
    from summarize_circuit_controls import paired_effects
    raw = np.array([[[10.], [10.], [7.], [5.], [8.]]])
    actual = paired_effects(raw)
    assert actual['reconstruction_shift'].item() == 3
    assert actual['unmatched_removal'].item() == 5
    assert actual['matched_removal'].item() == 2
    assert actual['residual_preserving_removal'].item() == 2
    raw[0, 1] = 9
    with pytest.raises(ValueError, match='No-op'):
        paired_effects(raw)


def test_bibliography_rejects_markup_duplicates_and_missing_citations():
    from audit_bibliography import validate_bibliography
    bib = '@article{alpha, title={A}}\n@article{beta, title={B}}'
    source = r'\cite{alpha,beta} and \citep[see][p. 2]{alpha}'
    assert validate_bibliography(bib, [source])['cited_keys'] == ['alpha', 'beta']
    with pytest.raises(ValueError, match='markup'):
        validate_bibliography(bib+'<? mode longmeta?>', [source])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_bibliography(bib+bib, [source])
    with pytest.raises(ValueError, match='Unresolved'):
        validate_bibliography(bib, [r'\cite{missing}'])


def test_main_only_preserves_main_figures_and_resolves_supplement_numbers():
    from build_main_only import make_main_source
    def figure(label):
        return r'\begin{figure}[htbp]\label{' + label + r'} Alt text: fixture\end{figure}'
    prefix = r'\begin{document} See ED \ref{ed11} and main \ref{fig1}.'
    prefix += ''.join(figure(f'fig{i}') for i in range(1, 5))
    source = prefix + r'\section*{Extended Data}'
    source += ''.join(figure(f'ed{i}') for i in range(1, 12)) + r'\end{document}'
    main, report = make_main_source(source)
    assert r'See ED 11 and main \ref{fig1}' in main
    assert main.count(r'\begin{figure}') == 4
    assert r'\label{ed1}' not in main
    assert report['replaced_reference_counts'] == {'ed11': 1}
    with pytest.raises(ValueError, match='Unknown supplementary'):
        make_main_source(source.replace(r'\ref{ed11}', r'\ref{ed99}'))
    with pytest.raises(ValueError, match='four ordered'):
        make_main_source(source.replace(figure('fig4'), ''))


def test_archived_steering_control_excludes_targets_not_full_category():
    from audit_steering_controls import archived_directions
    weights = np.random.default_rng(2288).normal(size=(8, 64)).astype(np.float32)
    enhanced = list(range(64))
    targets, sampled, vectors = archived_directions(weights, enhanced)
    assert not set(targets)&set(sampled)
    assert set(sampled) <= set(enhanced)
    assert len(sampled) == 20
    assert sampled == archived_directions(weights, enhanced)[1]
    for vector in vectors.values():
        assert np.isclose(np.linalg.norm(vector), 1)
    assert abs(np.dot(vectors['real'], vectors['orthogonalized'])) < 1e-6
    with pytest.raises(ValueError, match='Degenerate'):
        archived_directions(np.zeros_like(weights), enhanced)


def test_circuit_readout_validation_checks_prevalence_support_and_metadata():
    from summarize_circuit_controls import validate_readout, validate_metadata
    validate_readout(np.array([0., 2.]), np.array([0., .3]), (2,), 10)
    with pytest.raises(ValueError, match='nonnegative'):
        validate_readout(np.array([np.nan]), np.array([.5]), (1,), 10)
    with pytest.raises(ValueError, match='residue count'):
        validate_readout(np.array([2.]), np.array([.33]), (1,), 10)
    with pytest.raises(ValueError, match='zero-support'):
        validate_readout(np.array([0.]), np.array([.5]), (1,), 10)
    with pytest.raises(ValueError, match='shape'):
        validate_readout(np.array([1.]), np.array([.5]), (2,), 10)
    record = {'protein': 'p', 'cluster': 'c', 'length': 2, 'pair': '1_2', 'conditions': ['normal']}
    validate_metadata(record, 'p', {'p': 'c'}, {'p': 'AA'}, '1_2', ['normal'])
    with pytest.raises(ValueError, match='protein metadata'):
        validate_metadata({**record, 'length': 3}, 'p', {'p': 'c'}, {'p': 'AA'})
    with pytest.raises(ValueError, match='layer-pair'):
        validate_metadata(record, 'p', {'p': 'c'}, {'p': 'AA'}, '1_3', ['normal'])


def test_circuit_report_rejects_partial_or_invalid_protocol():
    from copy import deepcopy
    from build_circuit_report import validate_protocol, CONTRASTS
    summary = {'verified': True, 'n_discovery_clusters': 50,
               'n_evaluation_clusters': 100, 'bootstrap_resamples': 1000,
               'pairs': {'16_24': {'tested_edges': 5000,
                   'decomposition_max_abs_error': 0.,
                   'readout_summary': {name: {'mean_absolute_change': 1.,
                       'conditional_95_percentile_interval': [.5, 1.5]} for name in CONTRASTS}}}}
    identity = {'model': 'esm2', 'upstream_count': 100, 'downstream_count': 50}
    validate_protocol(summary, identity, 'esm2')
    for field, value in [('verified', False), ('n_evaluation_clusters', 2),
                         ('bootstrap_resamples', 100)]:
        with pytest.raises(ValueError, match='full-cohort'):
            validate_protocol({**summary, field: value}, identity, 'esm2')
    broken = deepcopy(summary)
    broken['pairs']['16_24']['readout_summary']['matched_removal']['conditional_95_percentile_interval'] = [2., 1.]
    with pytest.raises(ValueError, match='interval'):
        validate_protocol(broken, identity, 'esm2')
    broken = deepcopy(summary)
    del broken['pairs']['16_24']['readout_summary']['matched_removal']
    with pytest.raises(ValueError, match='contrasts'):
        validate_protocol(broken, identity, 'esm2')


def test_circuit_report_detects_changed_artifacts(tmp_path):
    from build_circuit_report import check_digest
    from probe_checkpoint import file_identity
    path = tmp_path/'effect.bin'
    path.write_bytes(b'paired effect')
    record = file_identity(path, content=True)
    check_digest(record)
    path.write_bytes(b'changed value')
    with pytest.raises(ValueError, match='Changed verified artifact'):
        check_digest(record)


def test_steering_directions_use_independent_streams_and_corrected_pool():
    from steering_controls import make_directions, condition_order
    weights = np.random.default_rng(42).normal(size=(32, 100)).astype(np.float32)
    categories = {i: 'enhanced' if i < 20 else 'unclassified' if i < 80 else 'inactive' for i in range(100)}
    first, meta = make_directions(weights, list(range(20)), categories)
    second, again = make_directions(weights, list(range(20)), categories)
    assert meta == again and len(first) == 16
    assert len(condition_order(first)) == 146
    for name, vector in first.items():
        assert np.array_equal(vector, second[name])
        assert np.linalg.norm(vector) == pytest.approx(1., abs=2e-7)
        if name.startswith('decoder'):
            assert len(set(meta[name]['feature_ids'])) == 20
            assert all(20 <= i < 80 for i in meta[name]['feature_ids'])
        if name.startswith('orthogonal'):
            assert abs(np.dot(vector, first['target'])) < 2e-7
            seed = name.split('_')[1]
            assert abs(np.dot(vector, first[f'random_{seed}'])) < .9
    with pytest.raises(ValueError, match='seeds'):
        make_directions(weights, list(range(20)), categories, seeds=(1, 1))
    with pytest.raises(ValueError, match='Insufficient'):
        make_directions(weights, list(range(20)), {i: 'enhanced' for i in range(100)})


def test_steering_hook_preserves_metadata_and_cleans_up_on_failure():
    import torch
    from steering_controls import add_direction, steering_intervention
    class Block(torch.nn.Module):
        def forward(self, value):
            return value, 'metadata'
    block = Block()
    original = torch.ones((1, 4, 3), dtype=torch.bfloat16)
    direction = np.array([1., 0., 0.], dtype=np.float32)
    assert add_direction(original, direction, 0.) is original
    with steering_intervention(block, direction, 2.):
        result, meta = block(original)
        assert meta == 'metadata'
        assert result.dtype == torch.float32
        assert torch.equal(result, original.float()+torch.tensor([2., 0., 0.]))
    assert block(original)[0] is original
    with pytest.raises(RuntimeError, match='test failure'):
        with steering_intervention(block, direction, 2.):
            raise RuntimeError('test failure')
    assert len(block._forward_hooks) == 0
    with pytest.raises(ValueError, match='shape'):
        add_direction(original, np.ones(4), 1.)
    with pytest.raises(ValueError, match='strength'):
        add_direction(original, direction, float('nan'))


def test_steering_readouts_use_aligned_residues_and_signed_activation_change():
    from summarize_steering_controls import readouts
    logits = np.array([[2., 0.], [0., 2.]])
    z = np.array([[1., 0.], [3., 2.]])
    assert np.array_equal(readouts(logits, logits, z, z, [0]), np.zeros(3))
    scores = readouts(logits, logits[:, ::-1], z, z+1, [0])
    assert scores[0] > 0 and scores[1] == 1 and scores[2] == 1
    assert readouts(logits, logits, z, z*.5, [0])[2] == -1
    with pytest.raises(ValueError, match='aligned'):
        readouts(logits, logits[:1], z, z, [0])
    with pytest.raises(ValueError, match='columns'):
        readouts(logits, logits, z, z, [0, 0])


def test_journal_conversion_preserves_caption_text_and_author_statements():
    import re
    from build_journal_manuscript import journal_source
    source = (Path(__file__).resolve().parents[1]/'revision/manuscript/sn-article.tex').read_text()
    generated, refs = journal_source(source)
    assert r'\documentclass[numsec,webpdf,modern,large]' in generated
    assert r'\bibliographystyle{oup-abbrvnat}' in generated
    assert r'\usepackage{oup-local-compat}' in generated
    assert generated.index(r'\revisionfitheader') < generated.index(r'\maketitle')
    assert r'\cite{' not in generated
    assert r'\ref{ed' not in generated
    assert r'\section*{Extended Data}' not in generated
    assert generated.count(r'\begin{figure*}') == 4
    assert generated.count('Alt text:') == 4
    for section in ('Funding', 'Competing interests'):
        match = re.search(r'\\section\*\{'+section+r'\}\s*([^\n]+)', source)
        assert match.group(1) in generated
    original = re.findall(r'\\begin\{figure\}(?:\[[^]]*\])?(.*?)\\end\{figure\}', source, re.S)[:4]
    converted = re.findall(r'\\begin\{figure\*\}(?:\[[^]]*\])?(.*?)\\end\{figure\*\}', generated, re.S)
    for before, after in zip(original, converted):
        assert before.split(r'\caption', 1)[1] == after.split(r'\caption', 1)[1] or (
            re.sub(r'\\ref\{ed(\d+)\}', r'\1', before.split(r'\caption', 1)[1]) == after.split(r'\caption', 1)[1])
    assert len(refs['main_figure_labels']) == 4


def test_journal_pdf_text_rejects_overflow_and_missing_alt_text():
    from verify_journal_manuscript import check_pdf_text
    xml = '<html><page width="100" height="100"><word xMin="1" yMin="1" xMax="90" yMax="10">Alt text: Alt text: Alt text: Alt text:</word></page></html>'
    assert check_pdf_text(xml) == 1
    with pytest.raises(ValueError, match='outside page bounds'):
        check_pdf_text(xml.replace('xMax="90"', 'xMax="110"'))
    with pytest.raises(ValueError, match='missing main figure alt text'):
        check_pdf_text(xml.replace('Alt text:', 'caption', 1))


def test_revised_heading_highlighting_against_submitted_baseline():
    from audit_revision_highlighting import check_headings
    baseline = '\\section{Results}\n\\subsection{Old claim}\n'
    with pytest.raises(ValueError, match='Unmarked revised heading'):
        check_headings(baseline, '\\subsection{New claim}\n')
    rows = check_headings(baseline, '\\section{Results}\n\\subsection{\\rev{New claim}}\n')
    assert [r['new_or_changed'] for r in rows] == [False, True]
    root = Path(__file__).resolve().parents[1]
    rows = check_headings((root/'revision/submitted/sn-article.tex').read_text(),
                          (root/'revision/manuscript/sn-article.tex').read_text())
    assert sum(r['new_or_changed'] for r in rows) == 7


def test_steering_figure_rejects_partial_cohorts_and_changed_means():
    from build_steering_results import validate_summary
    from steering_controls import condition_order, SEEDS
    from summarize_steering_controls import METRICS
    directions = {'target': {'kind': 'target'}}
    for seed in SEEDS:
        for kind in ('random', 'orthogonal', 'decoder'):
            directions[f'{kind}_{seed}'] = {'kind': kind}
    identity = {'directions': directions, 'conditions': [list(r) for r in condition_order(directions)]}
    summary = {'verified': True, 'n_proteins': 100, 'n_evaluation_clusters': 100,
               'bootstrap_resamples': 1000, 'conditions': {
                   name: {key: {'mean': 0.} for key in METRICS}
                   for name, _, _ in identity['conditions']}}
    scores = np.zeros((100, 146, 3))
    assert len(validate_summary(summary, identity, scores)) == 146
    with pytest.raises(ValueError, match='full-cohort'):
        validate_summary({**summary, 'n_proteins': 2}, identity, scores)
    summary['conditions']['target_alpha_50']['target_activation_change']['mean'] = 1.
    with pytest.raises(ValueError, match='Summary/score'):
        validate_summary(summary, identity, scores)
    summary['conditions']['target_alpha_50']['target_activation_change']['mean'] = 0.
    scores[0, 0, 0] = 1.
    with pytest.raises(ValueError, match='no-op'):
        validate_summary(summary, identity, scores)


def test_historical_empty_ablation_is_reconstruction_not_identity():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
    from models.interventions import sae_feature_ablation

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.Module()
            self.transformer.blocks = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, value):
            return self.transformer.blocks[0](value)

    class ImperfectSAE:
        def encode(self, value):
            return value.clone()

        def decode(self, value):
            return value * .5

    model = ToyModel()
    original = torch.ones((1, 3, 2))
    with sae_feature_ablation(model, 'esm3', ImperfectSAE(), 0, []):
        assert torch.equal(model(original), original * .5)
    assert torch.equal(model(original), original)


def test_streaming_digest_is_independent_of_chunk_boundaries(tmp_path):
    import hashlib
    path = tmp_path/'input.bin'
    content = b'fixed-input-data' * 37
    path.write_bytes(content)
    first = stable_digest(path, chunk_size=7)
    second = stable_digest(path, chunk_size=64)
    assert first == second
    assert first['sha256'] == hashlib.sha256(content).hexdigest()
    assert first['bytes'] == len(content)


def test_attention_marginal_metric_does_not_establish_query_pattern_identity():
    left = np.array([[.9, .1], [.1, .9]])
    right = left[::-1].copy()
    assert not np.array_equal(left, right)
    assert query_marginal_jsd(left, right) == 0
    assert query_marginal_jsd(np.array([[1., 0.]]), np.array([[0., 1.]])) == pytest.approx(np.log(2), abs=1e-8)
    with pytest.raises(ValueError, match='aligned'):
        query_marginal_jsd(left, right[:, :1])


def test_residue_alignment_joins_protein_ids_without_changing_positions():
    ids = np.array(['a', 'b', 'c']); offsets = np.array([[0, 2], [2, 3], [5, 1]])
    target_ids = np.array(['c', 'a', 'b']); target_offsets = np.array([[0, 1], [1, 2], [3, 3]])
    mapped, order = align_residue_rows(ids, offsets, target_ids, target_offsets, np.arange(6))
    assert mapped.tolist() == [1, 2, 3, 4, 5, 0]
    assert order.tolist() == [1, 2, 0]
    with pytest.raises(ValueError, match='lengths'):
        align_residue_rows(ids, offsets, target_ids, np.array([[0, 2], [2, 2], [4, 2]]), np.arange(6))
