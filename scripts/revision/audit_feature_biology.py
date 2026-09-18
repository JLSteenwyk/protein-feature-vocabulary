"""Audit descriptive ED4 inputs without new language-model calls or biology claims."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json, file_identity


def description_counts(records):
    counts = Counter(r['motif_type'] for r in records)
    unknown = [r for r in records if r['motif_type'] == 'unknown']
    json_like = sum(r['description'].lstrip().startswith(('```', '{', '[')) for r in unknown)
    return dict(counts), {'unknown': len(unknown), 'unknown_json_like': json_like,
                          'unknown_other': len(unknown)-json_like}


def join_weights(weights, categories):
    mapping = {r['feature_id']: r['category'] for r in categories}
    if len(mapping) != len(categories):
        raise ValueError('Duplicate category feature ID')
    if len({r['feature_id'] for r in weights}) != len(weights):
        raise ValueError('Duplicate probe feature ID')
    result = []
    for r in weights:
        if r['feature_id'] not in mapping or not np.isclose(abs(r['weight']), r['abs_weight']):
            raise ValueError('Invalid probe/category join')
        result.append({**r, 'category': mapping[r['feature_id']]})
    return result


def main():
    paths = [RESULTS/'residue_level_autointerpretability.json', RESULTS/'interpretable_probing.json',
             RESULTS/'per_feature_type_probing.json', RESULTS/'cross_modal_features.json',
             OUT/'cross_modal_complete.json']
    descriptions, probe, types, old_categories, categories = [json.loads(p.read_text()) for p in paths]
    records = descriptions['per_feature']
    counts, missing = description_counts(records)
    if len(records) != descriptions['n_features'] or counts != descriptions['motif_type_distribution']:
        raise ValueError('Archived description counts disagree')
    top = join_weights(probe['esm3']['top_features'][:25], categories['per_feature'])
    if any(a['abs_weight'] < b['abs_weight'] for a, b in zip(top, top[1:])):
        raise ValueError('Probe weights not ranked')
    old_enhanced = set(old_categories['enhanced_feature_ids'])
    sources = ['scripts/scaled_1.5M/23_residue_level_autointerpretability.py',
               'scripts/scaled_1.5M/18_per_feature_type_probing.py',
               'scripts/scaled_1.5M/09_interpretable_probing.py',
               'scripts/publication_figures/supplementary/ed04_feature_biology.py']
    report = {'n_selected_descriptions': len(records), 'motif_counts': counts, 'missingness': missing,
              'top25': top, 'top25_category_counts': dict(Counter(r['category'] for r in top)),
              'old_top25_enhanced': sum(r['feature_id'] in old_enhanced for r in top),
              'selection': 'First 50 archived enhanced IDs plus first 50 probe-weight IDs, deduplicated and eligibility-filtered; not representative dictionary sample.',
              'description_scope': 'First 15 selected high-activation contexts with annotation metadata supplied to LLM; no held-out prediction stage implemented. JSON-like unknown strings are not recovered classifications.',
              'composition_withdrawal': 'First feature with counts; top five amino acids among first 1000 active rows in storage order, no background-normalized preference or full-feature census.',
              'per_type_withdrawal': 'Disulfide start/end expanded to interval instead of two endpoints. Random accession-index train/test splits differ across model storage orders; validation randomly splits residues; no cluster separation. Raw test scores and sampled labels absent.',
              'weight_scope': 'Historical unstandardized L1 probe coefficients, not corrected-cluster probe coefficients or causal feature importance. Fixed ranking retained; complete revised zero-based category table used. No category enrichment test.',
              'inputs': [file_identity(p, content=True) for p in paths] +
                        [file_identity(ROOT/p, content=True) for p in sources],
              'script': file_identity(Path(__file__), content=True)}
    atomic_json(OUT/'feature_biology_audit.json', report)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    labels = ['Unknown: JSON-like text', 'Unknown: other text'] + [k for k in counts if k != 'unknown']
    values = [missing['unknown_json_like'], missing['unknown_other']] + [counts[k] for k in counts if k != 'unknown']
    axes[0].barh(range(len(labels)), values, color=['#999999', '#BBBBBB']+['#0072B2']*(len(labels)-2))
    axes[0].set(yticks=range(len(labels)), yticklabels=labels, xlabel='Selected feature descriptions',
                title='A  Archived labels, not validated concepts')
    axes[0].invert_yaxis()
    for i, v in enumerate(values):
        axes[0].text(v+.5, i, str(v), va='center')
    axes[0].set_xlim(0, max(values)+10)
    colors = {'enhanced': '#D55E00', 'suppressed': '#0072B2', 'unclassified': '#777777', 'inactive': '#CCCCCC'}
    axes[1].barh(range(len(top)), [r['abs_weight'] for r in top], color=[colors[r['category']] for r in top])
    axes[1].set(yticks=range(len(top)), yticklabels=[f"F{r['feature_id']}" for r in top],
                xlabel='Absolute historical probe coefficient', title='B  Corrected category join; fixed old ranking')
    axes[1].invert_yaxis()
    axes[1].legend(handles=[Patch(color=colors[c], label=c) for c in ['enhanced', 'unclassified']], loc='lower right')
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT/f'revision/figures/feature_biology_corrected.{ext}', dpi=220)
    plt.close(fig)
    print(json.dumps({'missingness': missing, 'top25_category_counts': report['top25_category_counts']}, indent=2))


if __name__ == '__main__':
    main()
