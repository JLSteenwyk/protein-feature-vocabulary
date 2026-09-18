"""Reconstruct archived steering-control inputs without new protein interventions."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json, file_identity

sys.path.insert(0, str(ROOT/'src'))


def archived_directions(weights, enhanced_ids, seed=42):
    weights = np.asarray(weights, dtype=np.float32)
    selected = list(enhanced_ids[:20])
    if weights.ndim != 2 or len(selected) != 20 or len(set(selected)) != 20:
        raise ValueError('Expected decoder matrix and twenty distinct target IDs')
    if min(selected) < 0 or max(selected) >= weights.shape[1] or not np.isfinite(weights).all():
        raise ValueError('Invalid decoder/target IDs')

    def unit(vector):
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Degenerate direction')
        return vector/norm

    real = unit(weights[:, selected].mean(axis=1))
    rng = np.random.RandomState(seed)
    random = unit(rng.randn(weights.shape[0]).astype(np.float32))
    # This reproduces the historical bug; it does not define a corrected control.
    pool = [i for i in range(weights.shape[1]) if i not in set(selected)]
    sampled = rng.choice(pool, 20, replace=False)
    shuffled = unit(weights[:, sampled].mean(axis=1))
    orthogonal = unit(random-np.dot(random, real)*real)
    return selected, sampled.tolist(), {'real': real, 'random': random,
                                       'shuffled_decoder': shuffled, 'orthogonalized': orthogonal}


def main():
    torch.set_num_threads(2)
    paths = {'script': ROOT/'scripts/scaled_1.5M/31_steering_controls.py',
             'main_script': ROOT/'scripts/scaled_1.5M/21_steering_vectors.py',
             'checkpoint': ROOT/'models/sae_1.5M/esm3_residue_ef8_k64/best.pt',
             'archived_categories': RESULTS/'cross_modal_features.json',
             'corrected_categories': OUT/'cross_modal_complete.json',
             'controls': RESULTS/'steering_controls.json', 'steering': RESULTS/'steering_vectors.json',
             'sequences': ROOT/'data/eval_expanded/sequences.json',
             'structures': ROOT/'data/eval_expanded/structures/manifest.json',
             'clusters': OUT/'clusters.json'}
    inputs = {name: file_identity(path, content=True) for name, path in paths.items()}
    ckpt = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    weights = ckpt['model_state_dict']['decoder.weight'].numpy()
    old = json.loads(paths['archived_categories'].read_text())
    corrected = json.loads(paths['corrected_categories'].read_text())
    if corrected['checkpoint_sha256'] != inputs['checkpoint']['sha256']:
        raise ValueError('Corrected categories use a different SAE checkpoint')
    categories = {row['feature_id']: row['category'] for row in corrected['per_feature']}
    selected, sampled, vectors = archived_directions(weights, old['enhanced_feature_ids'])
    legacy_main = np.mean([weights[:, i] for i in selected], axis=0)
    legacy_main /= np.linalg.norm(legacy_main)
    directions_path = OUT/'archived_steering_control_directions.npz'
    with directions_path.open('wb') as stream:
        np.savez_compressed(stream, **vectors)
    sequences = json.loads(paths['sequences'].read_text())
    structures = json.loads(paths['structures'].read_text())['available']
    clusters = json.loads(paths['clusters'].read_text())
    candidates = [(p, s) for p, s in sorted(sequences.items()) if 50 <= len(s) <= 500]
    selected_indices = np.random.RandomState(42).choice(len(candidates), min(200, len(candidates)), replace=False)
    control_ids = [candidates[i][0] for i in selected_indices]
    main_candidates = [p for p in sorted(structures) if p in sequences and len(sequences[p]) <= 500]
    main_ids = np.random.RandomState(42).choice(main_candidates, min(500, len(main_candidates)), replace=False).tolist()
    steering = json.loads(paths['steering'].read_text())
    retained_ids = [row['accession'] for row in steering['per_protein']]
    controls = json.loads(paths['controls'].read_text())
    if set(controls['per_direction']) != set(vectors):
        raise ValueError('Unexpected archived direction inventory')
    strengths = [-50., -20., -10., -5., 5., 10., 20., 50.]
    for direction, records in controls['per_direction'].items():
        if set(records) != {f'alpha_{a}' for a in strengths}:
            raise ValueError('Incomplete strength inventory')
        if any(r['n'] != 200 or not np.isfinite([r['mean_kl'], r['std_kl']]).all() for r in records.values()):
            raise ValueError('Unexpected archived counts or nonfinite aggregate')
    report = {'scope': 'Reconstructed inputs and archived aggregates only; no recovered paired observations or new biological steering effects.',
              'inputs': inputs, 'audit_script': file_identity(Path(__file__), content=True),
              'selected_target_ids': selected, 'sampled_decoder_control_ids': sampled,
              'target_corrected_categories': dict(Counter(categories[i] for i in selected)),
              'sampled_control_corrected_categories': dict(Counter(categories[i] for i in sampled)),
              'sampled_overlap_with_full_archived_enhanced': sorted(set(sampled)&set(old['enhanced_feature_ids'])),
              'sampled_overlap_with_twenty_targets': sorted(set(sampled)&set(selected)),
              'control_pool_excludes_all_archived_enhanced': False,
              'direction_norms': {k: float(np.linalg.norm(v)) for k, v in vectors.items()},
              'direction_cosines': {a: {b: float(np.dot(x, y)) for b, y in vectors.items()} for a, x in vectors.items()},
              'main_vs_control_target_vector_max_abs': float(np.abs(legacy_main-vectors['real']).max()),
              'directions_export': file_identity(directions_path, content=True),
              'cohort': {'candidate_count': len(candidates), 'control_selected_ids': control_ids,
                         'control_detected_clusters': len({clusters[p] for p in control_ids}),
                         'main_selected_ids': main_ids, 'retained_main_record_ids': retained_ids,
                         'control_overlap_with_main_selection': len(set(control_ids)&set(main_ids)),
                         'control_overlap_with_retained_main_records': len(set(control_ids)&set(retained_ids)),
                         'main_retained_records_match_selected_prefix': main_ids[:len(retained_ids)] == retained_ids},
              'archived_aggregates': controls,
              'limitations': ['Control archive has means, population SD and counts only; no per-protein IDs or paired observations.',
                              'Selected IDs reconstructed from current archived inputs do not establish historical successful-row identity.',
                              'One seeded vector per control type; orthogonalized control is derived from the same random vector.',
                              'Decoder sampling is not matched for activation prevalence, decoder geometry or feature category.',
                              'Main 500-protein and control 200-protein sampling pools differ; do not join them as paired observations.',
                              'Only KL is saved for controls; selected encoder rows are unused, so no target-activation specificity control was measured.',
                              'No alpha=0 condition or raw logits; no calibrated paired uncertainty can be reconstructed.',
                              'Historical hook removal is not protected by finally; exceptions could leave hooks installed, but the archive alone does not establish that this occurred.'],
              'feasibility': 'Cached model/checkpoint and generic human cohort are available; paired controls with repeated seeds and full records are feasible after current GPU jobs.'}
    atomic_json(OUT/'steering_control_audit.json', report)
    print(json.dumps({k: report[k] for k in ['target_corrected_categories', 'sampled_control_corrected_categories',
                                           'sampled_overlap_with_full_archived_enhanced', 'main_vs_control_target_vector_max_abs']}, indent=2))
    print('random/orthogonal cosine:', report['direction_cosines']['random']['orthogonalized'])
    print('cohort overlap:', report['cohort']['control_overlap_with_main_selection'])


if __name__ == '__main__':
    main()
