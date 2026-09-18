"""Summarize recoverable archived intervention variation without new interventions."""
import json
import numpy as np
from common import RESULTS, OUT, SEED, save


def main():
    steering = json.loads((RESULTS/'steering_vectors.json').read_text())
    clusters = json.loads((OUT/'clusters.json').read_text())
    rows = steering['per_protein']
    ids = [r['accession'] for r in rows]
    assert len(set(ids)) == len(ids)
    groups, gi = np.unique([clusters[p] for p in ids], return_inverse=True)
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=2000)
    metrics = ['mean_kl', 'target_act_change', 'token_change_rate']
    conditions = {}
    for alpha in steering['strengths']:
        key = f'alpha_{alpha}'
        present = np.array([key in r for r in rows])
        counts = np.bincount(gi[present], minlength=len(groups))
        samples = np.array([[r[key][metric] for metric in metrics] for r in rows if key in r])
        sums = np.zeros((len(groups), len(metrics)))
        np.add.at(sums, gi[present], samples)
        denominators = weights @ counts
        valid = denominators > 0
        boot = (weights[valid] @ sums)/denominators[valid, None]
        conditions[key] = {'n_archived_records': int(present.sum()), 'metrics': {}}
        for j, metric in enumerate(metrics):
            conditions[key]['metrics'][metric] = {'subset_mean': float(samples[:, j].mean()),
                                                  'subset_sd_ddof1': float(samples[:, j].std(ddof=1)),
                                                  'cluster_percentile_ci95': np.quantile(boot[:, j], [.025, .975]).tolist()}
    heads = json.loads((RESULTS/'head_ablation_sst.json').read_text())
    head_rows = []
    for condition in ['condition_s', 'condition_sst']:
        for head, record in heads[condition]['per_head'].items():
            head_rows.append({'condition': condition, 'head': head, 'mean_kl': record['mean_kl'],
                              'archived_sd_kl': record['std_kl']})
    result = {'steering': {'reported_full_n': steering['n_proteins'], 'archived_n': len(rows),
                          'archived_cluster_n': len(groups), 'seed': SEED, 'bootstrap_resamples': 2000,
                          'scope': 'Intervals only for the retained first 100 protein records, conditional on existing '
                                   'fitted model and intervention settings. Not intervals for all 500 proteins; '
                                   'not an independently sampled validation cohort.',
                          'full_aggregate_summary': steering['summary'], 'archived_subset': conditions},
              'head_ablation': {'n_proteins_reported': heads['n_proteins'], 'tested_layers': heads['ablation_layers'],
                                'heads_per_layer': heads['n_heads'], 'records': head_rows,
                                'scope': 'Archived mean and standard deviation only; paired protein resampling is '
                                         'not reconstructible from these aggregates. No all-layer maximum claim.'},
              'ss3': {'status': 'withdrawn', 'reason': 'Historical [:3] logits select special tokens; '
                      'no corrected SS3 result can be derived from archived aggregate agreement.'}}
    save('archived_intervention_uncertainty.json', result)
    print('Summarized', len(rows), 'steering records and', len(head_rows), 'head-condition aggregates', flush=True)


if __name__ == '__main__':
    main()
