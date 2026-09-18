"""Trace historical figure inputs and quantify limits of archived mechanistic claims."""
import hashlib
import json
from common import ROOT, OUT
from probe_checkpoint import atomic_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scaled = ROOT/'results/scaled_1.5M'
    unified = ROOT/'results/unified'
    paths = [scaled/'attribution_patching.json', unified/'esm3/attribution_patching.json']
    same = json.loads((scaled/'same_protein_causal.json').read_text())
    output = {'attribution': {
        'esm3_scaled_and_unified_byte_identical': paths[0].read_bytes() == paths[1].read_bytes(),
        'sha256': {str(p.relative_to(ROOT)): digest(p) for p in paths},
        'figure_loader': 'scripts/publication_figures/fig3_multimodal.py',
        'cross_protein_implementation': 'scripts/unified/run_attribution_patching.py',
        'cross_protein_conditions': 'Both models receive sequence-only inputs; no ESM-3 structure tokens.',
        'loss': 'KL(source || target), summed over token positions and output vocabulary; special positions retained.',
        'alignment': 'Prefixes truncated to shorter token sequence, not a biological residue alignment.',
        'attribution': 'Absolute gradient-dot-hidden-difference sum; approximate local sensitivity, not direct patching measurement.',
        'same_protein_n': same['n_proteins_attribution'],
        'same_protein_implementation': 'scripts/scaled_1.5M/30_same_protein_causal.py',
        'same_protein_limitation': 'Different cohort and summed-token objective; similar layer profile does not prove cross-protein specificity.',
        'raw_pair_records_available_in_plotted_exports': False,
        'resolution': 'Withdraw early-layer structure-token explanation and specificity assertion. Retain only qualified descriptive sensitivity, not calibrated causal or per-residue comparison.'},
        'circuits': {'figure_loader': 'scripts/publication_figures/fig4_causal.py',
                     'implementation': 'scripts/unified/run_sparse_feature_circuits.py',
                     'intervention': 'src/models/interventions.py:sae_feature_ablation',
                     'checkpoint_population': 'models/sae/{esm2,esm3,esm2_scaled,esm3_scaled}; not the principal models/sae_1.5M dictionaries',
                     'baseline': 'Unmodified hidden state compared to decoded feature-ablated SAE reconstruction.',
                     'confounding': 'Even an empty feature-ablation list substitutes decode(encode(h)); reconstruction error is not held constant.',
                     'multiplicity': 'Bonferroni over 100 upstream candidates separately per downstream feature, not all 5,000 tested feature pairs.',
                     'effect_threshold': 'Median plus two SDs of observed upstream absolute mean effects; not a null distribution.',
                     'selection': 'Top features from first 100 sequences; first firing proteins used for tests; no independent feature-selection split.',
                     'raw_paired_protein_effects_available': False,
                     'resolution': 'Withdraw validated feature-specific circuit interpretation; keep counts only as historical reconstruction-confounded observations until matched-baseline reruns.',
                     'models': {}}}
    for model in ['esm3', 'esm2']:
        data = json.loads((unified/model/'attribution_patching.json').read_text())
        output['attribution'][model] = {'n_pairs': data['n_pairs'],
                                         'per_layer_sample_counts': sorted({r['n'] for r in data['per_layer'].values()}),
                                         'first_layer_mean': data['per_layer']['0']['mean_effect'],
                                         'last_layer_mean': data['per_layer'][str(data['n_layers']-1)]['mean_effect']}
        path = unified/model/'sparse_feature_circuits.json'
        circuits = json.loads(path.read_text())
        model_result = {'sha256': digest(path), 'n_candidate_proteins': circuits['n_proteins'], 'layer_pairs': {}}
        for name, pair in circuits['layer_pairs'].items():
            connections = pair['connections']
            model_result['layer_pairs'][name] = {
                'n_reported_connections': sum(r['n_significant_upstream'] for r in connections),
                'n_saved_connection_summaries': sum(len(r['upstream_connections']) for r in connections),
                'n_connected_downstream': len(connections),
                'n_selected_downstream': pair['n_downstream_features'],
                'firing_protein_counts': sorted({r['n_firing_proteins'] for r in connections}),
                'n_tested_upstream_per_downstream': pair['n_upstream_features']}
        output['circuits']['models'][model] = model_result
    atomic_json(OUT/'mechanistic_provenance_audit.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
