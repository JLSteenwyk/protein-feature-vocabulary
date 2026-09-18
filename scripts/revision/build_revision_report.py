"""Render revision results and figures from completed analysis checkpoints."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, OUT


def main():
    dest = ROOT / 'revision'
    figs = dest / 'figures'
    figs.mkdir(exist_ok=True)
    matching = json.loads((OUT / 'matching.json').read_text())
    go = json.loads((OUT / 'go_audit.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 200})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for key, label, color in [('many_to_one', 'Best match', '#0072B2'),
                              ('one_to_one', 'One-to-one', '#D55E00'),
                              ('mutual_nearest', 'Mutual neighbors', '#009E73')]:
        vals = matching[key]['fractions']
        axes[0].plot([float(k) for k in vals], list(vals.values()), 'o-', label=label, color=color)
    axes[0].set(xlabel='Correlation threshold (strict >)', ylabel='Fraction of all source features',
                title='A  Full evaluation (7,459 sources)', ylim=(0, 1))
    axes[0].legend(fontsize=8)
    keys = ['many_to_one', 'one_to_one']
    axes[1].bar([0, 1], [matching['heldout'][k]['fractions']['0.3'] for k in keys],
                color=['#0072B2', '#D55E00'])
    axes[1].set(xticks=[0, 1], xticklabels=['Best match\nn=7,267', 'One-to-one\nn=6,166'],
                ylabel='Fraction of selected pairs, r > 0.3', ylim=(0, 1),
                title='B  Frozen pairs on validation clusters')
    null = matching['search_adjusted_null']
    values = [v['fractions']['0.3'] for v in null['null_per_permutation']]
    axes[2].hist(values, bins=12, color='#999999')
    axes[2].axvline(null['observed_same_subset']['fractions']['0.3'], color='#0072B2', label='Observed subset')
    axes[2].set(xlabel='Best-match fraction, r > 0.3', ylabel='Permutations', xlim=(0, 1),
                title='C  Search-adjusted null (512 sources)')
    axes[2].legend(fontsize=8)
    for ext in ['png', 'pdf']:
        fig.savefig(figs / f'matching_controls.{ext}')
    plt.close(fig)

    cats = go['models']['esm3']['categories']
    names = ['enhanced', 'invariant', 'suppressed']
    display_names = ['enhanced', 'unclassified', 'suppressed']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    colors = ['#D55E00', '#777777', '#0072B2']
    for i, name in enumerate(names):
        c = cats[name]
        axes[0].bar(i, c['frac_enriched'], color=colors[i])
        lo, hi = c['feature_bootstrap_fraction_ci95']
        axes[0].errorbar(i, c['frac_enriched'], yerr=[[c['frac_enriched']-lo], [hi-c['frac_enriched']]],
                        color='black', capsize=3)
    axes[0].set(xticks=range(3), xticklabels=display_names, ylim=(0, 1.08), ylabel='Enriched / all category members',
                title='A  Complete feature records')
    axes[1].bar(range(3), [cats[n]['median_all_features'] for n in names], color=colors)
    axes[1].set(xticks=range(3), xticklabels=display_names, ylabel='Median enriched GO term count',
                title='B  All category members, including zeros')
    matched = go['models']['esm3']['study_size_matched']
    axes[2].bar([0, 1], [matched['median_enhanced'], matched['median_invariant']], color=colors[:2])
    axes[2].set(xticks=[0, 1], xticklabels=['enhanced', 'unclassified'], ylabel='Median enriched GO term count',
                title='C  Exploratory study-size matching')
    for ext in ['png', 'pdf']:
        fig.savefig(figs / f'go_categories_corrected.{ext}')
    plt.close(fig)

    lines = ['# Revision Analysis Results', '',
             'Generated from analysis checkpoints; not a declaration of submission readiness.', '',
             '## Corrected Circuit Controls', '',
             'Both full-cohort model evaluations are verified. The independently generated '
             '[circuit results](circuit_results.md) report all twelve paired contrasts, '
             'conditional intervals and links to complete exports. These replace neither '
             'the historical cohorts nor the withdrawn selected-edge graphs.', '',
             '## Paired Steering Controls', '',
             'The verified 100-cluster evaluation is integrated into main Figure 4. '
             '[Steering results](steering_results.md) link all strengths, individual '
             'seeds and paired contrasts. Decoder controls have much lower baseline '
             'prevalence than targets; these are not activity-matched controls or '
             'evidence of biological specificity.', '',
             '## Matching Controls', '',
             '| Definition | Fraction r > 0.3 | Median r | Number of pairs |',
             '| --- | ---: | ---: | ---: |']
    for key in ['many_to_one', 'reverse_many_to_one', 'one_to_one', 'mutual_nearest']:
        r = matching[key]
        lines.append(f"| {key} | {r['fractions']['0.3']:.4f} | {r['median_r']:.4f} | {r['n_pairs']} |")
    lines += ['', 'Mutual-neighbor fraction uses all 7,459 source features as denominator, not just mutual pairs.', '',
              f"Search-adjusted null: {null['n_permutations']} permutations; "
              f"observed subset fraction {null['observed_same_subset']['fractions']['0.3']:.4f}; "
              f"mean null {null['mean_null_fraction_gt_0.3']:.4f}; empirical p = {null['p_empirical']:.2f}.", '',
              '![Matching controls](figures/matching_controls.png)', '',
              '**Alt text:** Full-data matching fractions decline as the correlation threshold rises. '
              'One-to-one and mutual-neighbor criteria give lower coverage than best matching. '
              'Frozen discovery matches have lower coverage on validation clusters. '
              'Permuted protein rows give smaller best-match fractions than the observed subset.', '',
              'Held-out best-match fraction: 0.5002 (7,267 selected pairs). Held-out one-to-one '
              'fraction: 0.3163 (6,166 assigned pairs). Constant validation pairs count as failures. '
              'The matching bootstrap is conditional on discovery-selected pairs and fitted SAEs; '
              'its sparse threshold statistic has nonregular resampling behavior.', '',
              '## Corrected GO Categories', '',
              '| Category | Total | Testable | Enriched | Median all | Median enriched only |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, c in cats.items():
        name = {'invariant': 'unclassified', 'untested': 'inactive in paired set'}.get(name, name)
        lines.append(f"| {name} | {c['n_features']} | {c['n_testable']} | {c['n_enriched']} | "
                     f"{c['median_all_features']} | {c['median_enriched_only']} |")
    lines += ['', '![Corrected GO categories](figures/go_categories_corrected.png)', '',
              '**Alt text:** Most eligible features in all three categories have GO enrichment. '
              'Suppressed features have the highest median term count. The enhanced-versus-unclassified '
              'median difference disappears in an exploratory study-size-matched comparison.', '',
              'Error bars are descriptive feature-bootstrap intervals, not protein-level inference. '
              'Zero-valued untestable records mean no detected enrichment, not established absence '
              'of biological signal. The study-size-matched comparison is exploratory.', '',
              '## Functional-Site Controls', '']
    probe_path = OUT / 'probe_controls.json'
    if (OUT / 'combined_probe_controls.json').exists():
        probe_path = OUT / 'combined_probe_controls.json'
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        composition_path = OUT / 'composition_controls.json'
        if composition_path.exists() and 'composition' not in probe['models']:
            probe['models']['composition'] = json.loads(composition_path.read_text())
        count = sum(len(x) for x in probe['models'].values())
        lines += [f'Completed checkpoints: **{count}/26**. ' +
                  ('All planned fits saved.' if count == 26 else '**PARTIAL: do not interpret absent controls as null results.**'), '',
                  '| Partition | Proteins | Sampled residues | Positives | Prevalence |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for name, c in probe['sample_counts'].items():
            lines.append(f"| {name} | {c['proteins']} | {c['residues']} | {c['positive']} | {c['prevalence']:.5f} |")
        lines += ['', '| Model | Representation | AUROC | 95% cluster interval | Average precision | Within-protein AUROC | C |',
                  '| --- | --- | ---: | --- | ---: | ---: | ---: |']
        for model, results in probe['models'].items():
            for name, r in results.items():
                ci = r['ci95_auroc']
                lines.append(f"| {model} | {name} | {r['auroc']:.4f} | {ci['low']:.4f}-{ci['high']:.4f} | "
                             f"{r['average_precision']:.4f} | {r['macro_within_protein_auroc']:.4f} | {r['selected_C']} |")
                if r['convergence_warning']:
                    lines.append(f'\nWarning: {model}/{name} selected fit reached a convergence warning.\n')
        lines += ['', 'Within-protein AUROC is macro-averaged over proteins with both classes. '
                  'Intervals use 1,000 test-cluster percentile resamples, conditional on the fitted probe. '
                  'Unannotated residues are not verified biological negatives.']
    else:
        lines += ['No completed probe checkpoints yet.']
    cross_path = OUT / 'cross_modal_complete.json'
    reconstruction_path = OUT / 'reconstruction_resampled.json'
    if reconstruction_path.exists():
        reconstruction = json.loads(reconstruction_path.read_text())
        lines += ['', '## Matched Reconstruction Check', '',
                  f"Uniform matched sample: {reconstruction['n']:,} residues from {reconstruction['n_sampled_proteins']:,} "
                  f"proteins in {reconstruction['n_clusters']:,} detected clusters.", '',
                  '| Model | Global centered R2 | 95% cluster range | Mean cosine | Within-vector ratio |',
                  '| --- | ---: | --- | ---: | ---: |']
        for model, r in reconstruction['models'].items():
            low, high = r['r2_cluster_range95']
            lines.append(f"| {model} | {r['global_centered_r2']:.4f} | {low:.4f}-{high:.4f} | {r['mean_cosine']:.4f} | {r['mean_within_vector_variance_ratio']:.4f} |")
        lines += ['', reconstruction['scope'], '',
                  'The historical within-vector variance ratio is not global R2 and can ignore constant '
                  'within-vector reconstruction error. Historical residue samples used initial chunks '
                  'rather than uniform full-cohort positions. The revised ED1 distinguishes these metrics.', '',
                  '![Reconstruction definitions and matched evaluation](figures/reconstruction_corrected.png)', '',
                  '**Alt text:** Training curves have different absolute scales. TopK allows fewer than '
                  '64 positive activations. Historical cosine and within-vector ratios are distinct '
                  'from global centered R2 on matched residues, shown with conditional cluster ranges.']
    if cross_path.exists():
        cross = json.loads(cross_path.read_text())
        lines += ['', '## Complete Cross-Modal Recalculation', '',
                  f"Re-encoded {cross['n_proteins']:,} paired proteins in {cross['n_clusters']:,} detected clusters. "
                  f"All {cross['dictionary_width']:,} coordinates are exported; {cross['n_active']:,} activate "
                  f"in either condition and {cross['n_test_eligible']:,} have at least ten nonzero differences. "
                  'Protein summaries are residue means, not maxima. Ineligible active coordinates receive p=1 '
                  'and remain in the BH family.', '',
                  ', '.join(f'{k}: {v:,}' for k, v in cross['category_counts'].items()) + '.', '',
                  'Unclassified is not evidence of equivalence. Five hundred sequence-cluster resamples '
                  'provide conditional descriptive ranges of standardized differences. Complete records: '
                  '`analyses/cross_modal_complete.json`.']
        category_medians = {category: float(np.median([
            r['paired_standardized_difference_ddof0'] for r in cross['per_feature'] if r['category'] == category]))
            for category in ['enhanced', 'suppressed', 'unclassified']}
        lines += ['', 'Full-category median paired standardized differences: ' +
                  ', '.join(f'{k} {v:.3f}' for k, v in category_medians.items()) + '. '
                  'These replace selected-strongest-feature summaries; category separation is partly definitional.', '',
                  '![Complete cross-modal effects](figures/cross_modal_detail.png)', '',
                  '**Alt text:** Complete active-feature effect distributions, raw versus standardized changes, '
                  'and cluster-resampling ranges for fifteen fixed-rank feature examples. '
                  'Per-feature intervals are conditional, not simultaneous or independent validation of category selection.']
    balance_path = OUT / 'go_category_balance.json'
    if balance_path.exists():
        balance = json.loads(balance_path.read_text())
        b = balance['balance']['n_study_proteins']
        lines += ['', '## GO Join and Balance Verification', '',
                  f"Recomputed categories agree for all {balance['n_joined_sequence_active_features']:,} "
                  'sequence-active feature IDs after explicit renaming of invariant to unclassified and '
                  'untested to inactive in the paired set. No numerical category disagreement remains.', '',
                  f"Study-size log1p standardized mean difference: {b['before']['log1p_standardized_mean_difference']:.3f} "
                  f"before versus {b['after']['log1p_standardized_mean_difference']:.3f} after matching. "
                  f"{balance['n_exact_study_size_matches']:,}/{balance['n_pairs']:,} pairs have exactly equal study sizes. "
                  'Matched GO medians are 26 and 27. Remaining imbalance and dependence preclude a causal '
                  'interpretation. All pair IDs and activation-prevalence balance are saved in '
                  '`analyses/go_category_balance.json`.']
        if (figs/'go_detail_corrected.png').exists():
            lines += ['', '![Complete GO denominators and study-size dependence](figures/go_detail_corrected.png)', '',
                      '**Alt text:** Enrichment fractions use all category members; paired bars distinguish '
                      'all-member and enriched-only medians. Term counts increase with study size; matched '
                      'enhanced and unclassified distributions have medians 26 and 27.', '',
                      'This replaces Extended Data Figure 9. Former exploratory coactivation and DMS panels '
                      'are removed with provenance reasons in `figures/go_detail_provenance.json`; no corrected '
                      'DMS performance or functional interpretation of coactivation clusters is claimed.']
    export_path = OUT / 'go_complete_tests/manifest.json'
    if export_path.exists():
        exported = json.loads(export_path.read_text())
        lines += ['', 'Complete term-level exports: `analyses/go_complete_tests/`. NPZ arrays retain all '
                  'background terms for every active feature, with p/q, overlaps, eligibility, study membership, '
                  'annotation membership and identifiers. Ineligible coordinates have p=q=1.']
        for model, item in exported['models'].items():
            lines.append(f"- {model}: {item['n_features']:,} features by {item['n_terms']:,} terms; "
                         f"{item['count_mismatches']} count disagreements with the revised GO audit.")
    refit_path = OUT / 'probes_overlap_excluded/probe_controls.json'
    if refit_path.with_name('combined_probe_controls.json').exists():
        refit_path = refit_path.with_name('combined_probe_controls.json')
    lines += ['', '## Training-Homology-Excluded Probe Refits', '',
              'Separate from frozen-model test exclusion, these fits remove 844 flagged proteins before '
              'residue sampling and model selection. Original cluster assignments and fitted SAE dictionaries '
              'remain fixed. Outputs are isolated in `analyses/probes_overlap_excluded/`.']
    if refit_path.exists():
        refit = json.loads(refit_path.read_text())
        n_refit = sum(len(v) for v in refit['models'].values())
        lines.append(f'Completed refit checkpoints: **{n_refit}/26**. Partial unless all 26 are present.')
        lines += ['', '| Model | Representation | AUROC | Average precision | Within-protein AUROC |',
                  '| --- | --- | ---: | ---: | ---: |']
        for model, results in refit['models'].items():
            for name, r in results.items():
                lines.append(f"| {model} | {name} | {r['auroc']:.4f} | {r['average_precision']:.4f} | {r['macro_within_protein_auroc']:.4f} |")
    else:
        lines.append('No completed refit checkpoint yet; process status must be verified independently.')
    excluded_verification = OUT/'probes_overlap_excluded/probe_figure_verification.json'
    if excluded_verification.exists():
        lines += ['', 'All 26 exclusion prediction hashes, row alignment, class counts and point metrics '
                  'have been verified. The separate `exclusion_split_verification.json` reproduces every '
                  'sampled row and verifies removal before sampling with original partitions retained. '
                  'Complete checked values: `analyses/probes_overlap_excluded/probe_controls_verified.tsv`.']
    excluded_paired = OUT/'probes_overlap_excluded/combined_paired_probe_comparisons.json'
    if excluded_paired.exists():
        pairs = json.loads(excluded_paired.read_text())
        lines += ['', f'Exclusion paired comparisons saved: {len(pairs)}/27. Intervals condition on fitted '
                  'probes and the sampled test cohort; they are not SAE-training variability or '
                  'multiplicity-adjusted significance tests.']
        contrast = pairs.get('esm3_intact_vs_esm2_intact')
        if contrast:
            for metric, value in contrast['metrics'].items():
                lo, hi = value['ci95']
                lines.append(f"- ESM-3 minus ESM-2 {metric}: {value['difference']:.4f}; "
                             f"95% paired cluster interval [{lo:.4f}, {hi:.4f}].")
    paired_path = OUT / 'paired_probe_comparisons.json'
    if paired_path.with_name('combined_paired_probe_comparisons.json').exists():
        paired_path = paired_path.with_name('combined_paired_probe_comparisons.json')
    sensitivity_path = OUT / 'matching_sensitivity/summary.json'
    if sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text())
        lines += ['', '## Matching Sensitivities', '',
                  f"Completed {len(sensitivity['runs'])}/12 fixed-dictionary runs. Seeds 2288, 2289 and 2290 "
                  'define repeated cluster partitions, not independent biological replicates. '
                  'Exclusions precede discovery selection. One-to-one fractions below use assigned pairs; '
                  'breadth-restricted fractions use eligible sources, not all source features.', '',
                  '| Aggregation / cohort / seed | Best r>0.3 | One-to-one r>0.3 | Breadth-restricted r>0.3 | Breadth-eligible sources | Union-support r>0.3 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for name, r in sensitivity['runs'].items():
            restricted = r['breadth_eligible_source_denominator']
            lines.append(f"| {name} | {r['best']['fractions']['0.3']:.4f} | "
                         f"{r['one_to_one_assigned_denominator']['fractions']['0.3']:.4f} | "
                         f"{restricted['fractions']['0.3']:.4f} | {r['n_breadth_eligible_sources']} | "
                         f"{r['best_union_support']['fractions']['0.3']:.4f} |")
        lines += ['', 'Breadth matching requires at least ten activating discovery proteins for each '
                  'feature and a target activating count within a factor of two of the source. '
                  'Union-support correlations omit validation proteins where both frozen matched features '
                  'are zero; this changes the estimand and is not a replacement for the primary analysis. '
                  'Constant/undefined validation correlations remain failures. Full threshold curves, '
                  'all-source denominators, breadth strata and selected IDs are saved per run.', '',
                  '![Matching sensitivity](figures/matching_sensitivity.png)', '',
                  '**Alt text:** Three cluster splits yield similar coverage within each matching rule. '
                  'Mean aggregation gives higher coverage than maxima. Excluding detected training homologs '
                  'before selecting matches changes coverage little. Uniqueness and activation-breadth '
                  'restrictions give lower coverage than unrestricted best matching.']
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, field, title in zip(axes, ['protein_max_activations', 'protein_mean_activations'],
                                   ['A  Protein maxima', 'B  Protein means']):
            for ci, cohort in enumerate(['all', 'training_homologs_excluded']):
                for index, key in enumerate(['best', 'one_to_one_assigned_denominator', 'breadth_eligible_source_denominator']):
                    values = [r[key]['fractions']['0.3'] for r in sensitivity['runs'].values()
                              if r['aggregation'] == field and r['cohort'] == cohort]
                    positions = index + (ci-.5)*.22 + np.linspace(-.04, .04, len(values))
                    ax.scatter(positions, values, color=['#0072B2', '#D55E00'][ci],
                               marker=['o', '^'][ci], label=['All proteins', 'Homologs excluded'][ci] if index == 0 else None)
            ax.set(xticks=range(3), xticklabels=['Best\nall sources', 'One-to-one\nassigned pairs', 'Breadth\neligible sources'],
                   ylabel='Validation fraction r > 0.3', ylim=(0, .7), title=title)
            ax.legend(fontsize=8)
        for ext in ['png', 'pdf']:
            fig.savefig(figs/f'matching_sensitivity.{ext}')
        plt.close(fig)
    mechanism_path = OUT / 'mechanistic_provenance_audit.json'
    if mechanism_path.exists():
        lines += ['', '## Mechanistic Provenance Corrections', '',
                  'The ESM-3 scaled attribution file is byte-identical to the unified sequence-only result. '
                  'Both cross-protein models receive sequence-only inputs; the smaller ESM-3 early-layer '
                  'score cannot be explained by supplied structure tokens. Token-summed KL uses unaligned '
                  'prefixes, including special positions. The separate 196-protein modality contrast does '
                  'not establish specificity of the cross-protein comparison.', '',
                  'Historical circuit ablation replaces the whole hidden state with an SAE reconstruction. '
                  'Comparison with the original model confounds feature removal and reconstruction error. '
                  'An empty-ablation toy regression test demonstrates this. The archived counts '
                  '(235, 113, 224) reproduce, but their interpretation as validated feature-specific causal '
                  'edges is withdrawn. Bonferroni adjustment was per downstream feature, and the effect '
                  'threshold used observed effects, not an independent null. Full audit: '
                  '`analyses/mechanistic_provenance_audit.json`.']
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        lines += ['', '## Paired Probe Differences', '',
                  'Reference intact representation minus control; 500 shared test-cluster resamples. '
                  'These are conditional exploratory intervals, not multiplicity-adjusted tests. '
                  'Only completed prediction pairs are included; rerun the comparison script as fits finish.', '',
                  '| Comparison | AUROC difference (95% interval) | AP difference (95% interval) | Within-protein difference (95% interval) |',
                  '| --- | --- | --- | --- |']
        for name, result in paired.items():
            cells = []
            for key in ['auroc', 'average_precision', 'macro_within_protein_auroc']:
                m = result['metrics'][key]
                cells.append(f"{m['difference']:.4f} ({m['ci95'][0]:.4f}, {m['ci95'][1]:.4f})")
            lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    diagnostic = OUT / 'matching_bootstrap_diagnostic.json'
    if diagnostic.exists():
        d = json.loads(diagnostic.read_text())
        lines += ['', '## Matching Bootstrap Diagnostic', '',
                  f"Original fraction {d['original_fraction']:.5f}; bootstrap mean {d['bootstrap_mean']:.5f}. "
                  f"Average passing pairs lost to constant profiles: {d['mean_original_pass_lost_to_constant']:.3f}; "
                  f"passing pairs crossing below threshold while variable: {d['mean_original_pass_crossing_below_threshold_while_variable']:.3f}; "
                  f"failing pairs crossing above threshold: {d['mean_original_fail_crossing_above_threshold']:.3f}.", '',
                  'The shift is not solely due to rare-feature disappearance. A restriction to at least ten '
                  'active validation clusters does not remove it. The percentile range is a resampling '
                  'sensitivity description, not a calibrated confidence claim. Full pair IDs and '
                  'validation-support counts are saved for reproducibility.']
    dataset = OUT / 'dataset_audit.json'
    if dataset.exists():
        d = json.loads(dataset.read_text())
        lines += ['', '## Dataset Audit', '',
                  f"Exact matches to {d['n_training_fasta_records']:,} archived training FASTA records: "
                  f"{d['n_evaluation_with_exact_training_match']}. Saved probe partitions share no protein IDs, "
                  'detected sequence clusters, or exact sequences. Sampled residues belong to the stated partitions. '
                  'This does not exclude nonidentical training homologs or foundation-model pretraining overlap.']
    lines += ['', '## Main Figure Replacements', '',
              'The actual manuscript now uses `cross_modal_corrected.png` as Figure 3 and '
              '`steering_paired_corrected.png` as Figure 4. The invalid SS3 panel and '
              'reconstruction-confounded circuit-count panel are removed from these main figures. '
              'Source verification is recorded in `analyses/manuscript_figure_verification.json`; '
              'remaining historical figures are not thereby validated.', '',
              '![Revised cross-modal results](figures/cross_modal_corrected.png)', '',
              '**Alt text:** Most paired active features are unclassified. Enhanced and suppressed '
              'features occupy opposite effect tails. Raw GO category medians differ, but enhanced '
              'and unclassified medians are nearly equal after study-size matching.', '',
              '![Paired steering controls](figures/steering_paired_corrected.png)', '',
              '**Alt text:** Target and control perturbations change model outputs. Paired '
              'differences vary with strength and control type; some intervals cross zero. '
              'Target features activate much more frequently than sampled decoder controls.', '',
              '## Archived Lens Audit', '',
              'ED5 now reports agreement with each condition\'s own final predictions, not next-token '
              'or biological accuracy. S+St KL is lower only at sampled block 24. The tuned lens uses '
              'a separate 300/200 sequence-only train/evaluation cohort and is worse at the final block. '
              'Component norms are not commensurate causal contributions: ESM-3 hooks capture raw outputs '
              'before residual scaling and omit geometric attention. Missing individual records preclude '
              'reconstructed confidence intervals. See `analyses/lens_summary_audit.json`.', '',
              '![Descriptive archived lens summaries](figures/lens_descriptive.png)', '',
              '**Alt text:** Later intermediate predictions increasingly match final outputs. Condition '
              'and tuned-lens contrasts vary by block. Raw component norms use different measurement '
              'conventions between models and are not measures of causal importance.', '',
              '## Archived Contact Audit', '',
              'Revised ED8 retains descriptive contact ranks and group means only. Prediction budgets '
              'L/5, L/2 and L were mislabeled as separation thresholds; minimum separation is six. '
              'The supervised evaluation estimates L-5 rather than L and is withdrawn. The source '
              'reports 474 successful proteins but does not export per-protein predictions or per-head '
              'contributing counts. Coordinate alignment and missing-coordinate limitations remain. '
              'No corrected contact accuracy or biological uncertainty interval is inferred. See '
              '`analyses/contact_summary_audit.json`. Reconstruction-confounded circuit panels are removed.', '',
              '![Descriptive archived contact summaries](figures/contact_descriptive.png)', '',
              '**Alt text:** Late-block heads occupy the highest archived precision ranks. Atlas-low-JSD '
              'heads have higher group mean precision across three prediction budgets; this is not '
              'independent ranking validation or evidence of causal head specialization.', '',
              '## Reproduction', '', 'Run from the repository root:', '', '```bash',
              './env/bin/python scripts/revision/prepare_clusters.py',
              'OPENBLAS_NUM_THREADS=16 ./env/bin/python scripts/revision/run_matching.py',
              './env/bin/python scripts/revision/run_go_audit.py',
              './env/bin/python scripts/revision/recompute_cross_modal.py',
              './env/bin/python scripts/revision/export_go_tests.py',
              './env/bin/python scripts/revision/audit_go_category_balance.py',
              'OPENBLAS_NUM_THREADS=8 ./env/bin/python scripts/revision/matching_sensitivity.py',
              './env/bin/python scripts/revision/verify_matching_sensitivity.py',
              './env/bin/python scripts/revision/audit_mechanistic_provenance.py',
              './env/bin/python scripts/revision/audit_contact_summary.py',
              './env/bin/python scripts/revision/audit_lens_summary.py',
              'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 ./env/bin/python scripts/revision/recompute_reconstruction.py',
              './env/bin/python scripts/revision/verify_reconstruction.py',
              './env/bin/python scripts/revision/build_reconstruction_detail.py',
              './env/bin/python scripts/revision/build_cross_modal_detail.py',
              './env/bin/python scripts/revision/build_go_detail.py',
              './env/bin/python scripts/revision/combine_probe_runs.py',
              './env/bin/python scripts/revision/combine_probe_runs.py --base-run-dir revision/analyses/probes_overlap_excluded --continuation-dir revision/analyses/probes_overlap_excluded_esm2_aligned',
              'OPENBLAS_NUM_THREADS=2 ./env/bin/python scripts/revision/paired_probe_comparisons.py --combined',
              'OPENBLAS_NUM_THREADS=2 ./env/bin/python scripts/revision/paired_probe_comparisons.py --combined --run-dir revision/analyses/probes_overlap_excluded',
              './env/bin/python scripts/revision/audit_dataset.py',
              './env/bin/python scripts/revision/audit_training_homology.py',
              'OPENBLAS_NUM_THREADS=8 ./env/bin/python scripts/revision/diagnose_matching_bootstrap.py',
              './env/bin/python scripts/revision/build_revision_report.py',
              './env/bin/python scripts/revision/build_corrected_main_figures.py',
              './env/bin/python scripts/revision/verify_manuscript_figures.py',
              './env/bin/python -m pytest tests/test_revision_statistics.py -q', '```', '',
              'Do not restart an active job. New probe runs use an exclusive writer lock, atomic checkpoints, '
              'input/protocol identities and verified prediction alignment before skipping completed fits. '
              'Legacy checkpoints without an identity are deliberately not adopted or overwritten. '
              'Large input identities currently use path, size and modification time, not content hashes; '
              'release-wide checksums remain a separate requirement. Reports prefer validated combined snapshots '
              'without overwriting original runs. Refresh snapshots after new commits. See reproduction.md '
              'for the isolated ESM-2 alignment continuations and the original runner limitation.', '']
    (dest / 'analysis_results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
