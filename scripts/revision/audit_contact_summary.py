"""Audit archived contact summaries; no model inference or historical rewrites."""
import ast
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT
from probe_checkpoint import atomic_json, file_identity


def eligible_pair_count(length, min_separation=6):
    remaining = max(length-min_separation, 0)
    return remaining*(remaining+1)//2


def historical_estimated_length(n_pairs):
    return int((1+np.sqrt(1+8*n_pairs))/2)


def load_historical_precision():
    """Load only the archived pure metric functions, avoiding inference imports/main."""
    path = ROOT/'scripts/scaled_1.5M/26_contact_map_from_attention.py'
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ['apc_correction', 'contact_precision']]
    if len(nodes) != 2:
        raise ValueError('Historical metric definitions changed')
    namespace = {'np': np, 'MIN_SEQ_SEP': 6}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['contact_precision']


def main():
    source = ROOT/'results/scaled_1.5M/contact_map_from_attention.json'
    atlas_path = ROOT/'results/scaled_1.5M/attention_atlas.json'
    data, atlas = [json.loads(p.read_text()) for p in [source, atlas_path]]
    jsd = np.asarray(atlas['mean_jsd_matrix'])
    if jsd.shape != (48, 24) or np.count_nonzero(jsd > .1) != data['n_responsive_heads']:
        raise ValueError('Atlas head dimensions or responsive counts disagree')
    heads = data['top_heads']
    if len({(h['layer'], h['head']) for h in heads}) != len(heads):
        raise ValueError('Duplicate ranked heads')
    scores = [h['precision_L'] for h in heads]
    if not np.isfinite(scores).all() or not np.all((np.array(scores) >= 0) & (np.array(scores) <= 1)):
        raise ValueError('Invalid precision')
    if not np.all(np.diff(scores) <= 0):
        raise ValueError('Ranked head order inconsistent')
    for head in heads:
        if bool(jsd[head['layer'], head['head']] > .1) != head['is_responsive']:
            raise ValueError('Head category disagrees with atlas')
    budget_errors = [{'true_length': length, 'eligible_pairs': eligible_pair_count(length),
                      'historical_length_estimate': historical_estimated_length(eligible_pair_count(length))}
                     for length in [80, 100, 200, 300]]
    if any(r['historical_length_estimate'] != r['true_length']-5 for r in budget_errors):
        raise ValueError('Supervised length diagnostic failed')
    report = {
        'inputs': [file_identity(p, content=True) for p in [source, atlas_path,
            ROOT/'scripts/scaled_1.5M/26_contact_map_from_attention.py',
            ROOT/'scripts/publication_figures/supplementary/ed08_contact_circuits.py',
            OUT/'mechanistic_provenance_audit.json']],
        'reported_successful_proteins': data['n_proteins'], 'selected_cap': 500,
        'candidate_lengths': [80, 300], 'selection_seed': 42,
        'input_condition': 'sequence only', 'minimum_sequence_separation': 6,
        'distance_definition': 'CB distance <8 Angstrom; implementation falls back to CA whenever CB absent, not only glycine.',
        'ranking': 'Symmetrized residue attention, APC, upper-triangle pairs; L/5,L/2,L are prediction budgets.',
        'coordinate_limitations': 'PDB residue number minus one, no explicit sequence/chain/insertion-code alignment; missing coordinates treated as noncontacts, not masked.',
        'supervised_length_diagnostic': budget_errors,
        'supervised_resolution': 'Withdraw precision claims: excluded near-diagonal pairs imply estimated length L-5; scores/labels and exact protein lengths not archived for rescoring.',
        'supervised_split': 'First 70% of successful proteins train, remainder test; no homology-cluster split or saved membership.',
        'saved_top_heads': len(heads), 'top25_blocks': sorted({h['layer'] for h in heads[:25]}),
        'top_head': heads[0], 'precision_summary': data['precision_summary'],
        'per_protein_records_in_export': False,
        'uncertainty_limitations': 'No per-protein/head records, contributing counts or predictions exported; no valid cluster intervals can be recovered. Heads and head-protein pairs are not independent replicates.',
        'resolution': 'Retain descriptive archived ranks/group means only; not held-out ranking validation, causal structure-input specialization or corrected contact benchmark.',
        'withdrawn_ED8_panels': ['Supervised coefficients as evidence of a validated predictor', 'Reconstruction-confounded circuit counts'],
    }
    atomic_json(OUT/'contact_summary_audit.json', report)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), constrained_layout=True)
    top = heads[:15]
    axes[0].barh(range(len(top)), [h['precision_L'] for h in top], color='#0072B2')
    axes[0].set(yticks=range(len(top)), yticklabels=[f'L{h["layer"]}H{h["head"]}' for h in top],
                xlabel='Archived mean precision among top L pairs', xlim=(0, 1), title='A  Highest-ranked 15 heads (descriptive)')
    axes[0].invert_yaxis()
    for offset, key, label, color in [(-.18, 'responsive', 'JSD > 0.1 (28 heads)', '#D55E00'),
                                     (.18, 'non_responsive', 'JSD <= 0.1 (1,124 heads)', '#777777')]:
        values = [data['precision_summary'][k][key] for k in ['L/5', 'L/2', 'L']]
        axes[1].bar(np.arange(3)+offset, values, .34, color=color, label=label)
    axes[1].set(xticks=range(3), xticklabels=['floor(L/5)', 'floor(L/2)', 'L'],
                xlabel='Number of top-ranked pairs (not sequence separation)',
                ylabel='Archived mean precision', ylim=(0, .12), title='B  Atlas-defined head groups')
    axes[1].legend(fontsize=8)
    destination = ROOT/'revision/figures'
    for extension in ['png', 'pdf']:
        fig.savefig(destination/f'contact_descriptive.{extension}', dpi=220)
    plt.close(fig)
    print('Archived contact ranks verified; supervised length bug and uncertainty limits recorded.')


if __name__ == '__main__':
    main()
