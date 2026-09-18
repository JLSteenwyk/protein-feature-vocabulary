"""Verify recomputed category joins and expose balance of exploratory GO matching."""
import json
import numpy as np
from scipy.stats import spearmanr
from common import OUT, save


def balance(left, right, key):
    a = np.array([r[key] for r in left], dtype=float)
    b = np.array([r[key] for r in right], dtype=float)
    x, y = np.log1p(a), np.log1p(b)
    scale = np.sqrt((x.var(ddof=1) + y.var(ddof=1))/2)
    return {'median_enhanced': float(np.median(a)), 'median_unclassified': float(np.median(b)),
            'log1p_standardized_mean_difference': float((x.mean()-y.mean())/scale) if scale else None,
            'max_absolute_paired_log1p_gap': float(np.max(np.abs(x-y))) if len(x) == len(y) else None}


def main():
    go = json.loads((OUT/'go_audit.json').read_text())
    cross = json.loads((OUT/'cross_modal_complete.json').read_text())
    canonical = {r['feature_id']: r for r in cross['per_feature']}
    if len(canonical) != cross['dictionary_width'] or set(canonical) != set(range(cross['dictionary_width'])):
        raise ValueError('Cross-modal table must cover each dictionary coordinate exactly once')
    rows = go['models']['esm3']['per_feature']
    recoding = {'invariant': 'unclassified', 'untested': 'inactive'}
    disagreements = [r['feature_id'] for r in rows
                     if recoding.get(r['category'], r['category']) != canonical[r['feature_id']]['category']]
    if disagreements:
        raise ValueError(f'GO categories changed: {disagreements}')
    enhanced = [r for r in rows if canonical[r['feature_id']]['category'] == 'enhanced']
    unclassified = [r for r in rows if canonical[r['feature_id']]['category'] == 'unclassified']
    remaining = unclassified.copy()
    left, right = [], []
    for row in sorted(enhanced, key=lambda r: r['n_study_proteins']):
        j = min(range(len(remaining)), key=lambda j: abs(np.log1p(remaining[j]['n_study_proteins'])-
                                                         np.log1p(row['n_study_proteins'])))
        left.append(row); right.append(remaining.pop(j))
    pairs = [{'enhanced_feature_id': a['feature_id'], 'unclassified_feature_id': b['feature_id'],
              'enhanced_study_size': a['n_study_proteins'], 'unclassified_study_size': b['n_study_proteins']}
             for a, b in zip(left, right)]
    output = {'category_disagreements': disagreements, 'n_joined_sequence_active_features': len(rows),
              'inactive_in_paired_set_ids': [r['feature_id'] for r in rows if not canonical[r['feature_id']]['active']],
              'n_pairs': len(pairs), 'pair_selection': 'Greedy log1p study-size distance without replacement; ascending enhanced study size',
              'balance': {key: {'before': balance(enhanced, unclassified, key), 'after': balance(left, right, key)}
                          for key in ['n_study_proteins', 'n_nonzero_proteins']},
              'outcome': {'median_enhanced': float(np.median([r['n_terms_all_background_bh'] for r in left])),
                          'median_unclassified': float(np.median([r['n_terms_all_background_bh'] for r in right]))},
              'n_exact_study_size_matches': sum(a['n_study_proteins'] == b['n_study_proteins'] for a, b in zip(left, right)),
              'correlations': {}, 'pairs': pairs,
              'scope': 'Exploratory dictionary-feature balance; not a causal estimate or independent-protein significance test. '
                       'Log1p balance, support mismatch, annotation redundancy and feature dependence remain relevant.'}
    for category, cr in [('enhanced', enhanced), ('unclassified', unclassified)]:
        output['correlations'][category] = {
            key: float(spearmanr([r[key] for r in cr], [r['n_terms_all_background_bh'] for r in cr]).statistic)
            for key in ['n_study_proteins', 'n_nonzero_proteins']}
    save('go_category_balance.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'pairs'}, indent=2))


if __name__ == '__main__':
    main()
