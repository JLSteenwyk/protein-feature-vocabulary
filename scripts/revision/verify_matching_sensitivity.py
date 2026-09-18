"""Check saved split, exclusion, matching and denominator invariants for all repeats."""
import json
import numpy as np
from common import OUT, save
from run_matching import describe


def main():
    directory = OUT/'matching_sensitivity'
    summary = json.loads((directory/'summary.json').read_text())
    clusters = json.loads((OUT/'clusters.json').read_text())
    excluded = set(json.loads((OUT/'training_homology_audit.json').read_text())['evaluation_ids_identity50'])
    assert len(summary['runs']) == 12
    result = {'runs': {}, 'scope': 'Artifact-level split, exclusion, candidate eligibility, assignment uniqueness and fraction checks; not independent model inference.'}
    for name, record in summary['runs'].items():
        with np.load(directory/f'{name}.npz') as data:
            discovery, validation = set(data['discovery_ids']), set(data['validation_ids'])
            assert not discovery & validation
            assert not {clusters[p] for p in discovery} & {clusters[p] for p in validation}
            if record['cohort'] == 'training_homologs_excluded':
                assert not (discovery | validation) & excluded
                assert len(discovery | validation) == 11647
            else:
                assert len(discovery | validation) == 12491
            assert len(np.unique(data['one_to_one_source_ids'])) == len(data['one_to_one_source_ids'])
            assert len(np.unique(data['one_to_one_target_ids'])) == len(data['one_to_one_target_ids'])
            eligible = data['breadth_target_ids'] >= 0
            assert np.array_equal(eligible, data['breadth_n_candidates'] > 0)
            pos = np.searchsorted(data['target_ids'], data['breadth_target_ids'][eligible])
            sc, tc = data['source_discovery_active_counts'][eligible], data['target_discovery_active_counts'][pos]
            assert np.all((sc >= 10) & (tc >= 10) & (tc >= sc/2) & (tc <= sc*2))
            assert describe(data['best_validation_r']) == record['best']
            assert describe(data['one_to_one_validation_r']) == record['one_to_one_assigned_denominator']
            assert describe(data['breadth_validation_r']) == record['breadth_all_source_denominator']
            assert np.isnan(data['breadth_validation_r'][~eligible]).all()
            result['runs'][name] = {'passed': True, 'n_discovery': len(discovery), 'n_validation': len(validation)}
    reference = json.loads((OUT/'matching.json').read_text())['heldout']
    reproduced = summary['runs']['protein_max_activations_all_2288']
    assert np.isclose(reference['many_to_one']['fractions']['0.3'], reproduced['best']['fractions']['0.3'], atol=1e-12)
    assert np.isclose(reference['one_to_one']['fractions']['0.3'], reproduced['one_to_one_assigned_denominator']['fractions']['0.3'], atol=1e-12)
    result['original_seed_2288_passing_fractions_reproduced'] = True
    save('matching_sensitivity_verification.json', result)
    print('All 12 sensitivity artifacts verified; original seed passing fractions reproduced.')


if __name__ == '__main__':
    main()
