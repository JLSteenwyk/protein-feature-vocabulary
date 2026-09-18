"""Audit loss of information in historical case-study plotting, not protein function."""
import json
from pathlib import Path
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json, file_identity


def support_summary(records):
    totals = {'features': 0, 'truncated_features': 0, 'saved_positions': 0, 'reported_active_positions': 0}
    for r in records:
        saved = r['active_positions']
        n = r['n_active_residues']
        if len(saved) != len(set(saved)) or len(saved) > n:
            raise ValueError('Inconsistent saved activation support')
        totals['features'] += 1
        totals['truncated_features'] += int(len(saved) < n)
        totals['saved_positions'] += len(saved)
        totals['reported_active_positions'] += n
    return totals


def main():
    source = RESULTS/'case_study_proteins.json'
    cases = json.loads(source.read_text())['case_studies']
    # The historical generator chooses these storage indices, not a prespecified random sample.
    selected = [cases[i] for i in [2, 1, 3]]
    summary = support_summary([f for case in selected for f in case['top_features'][:10]])
    report = {'plotted_feature_support': summary,
              'heatmap_readout': 'Per-feature mean of positive activations copied to first at most 20 active positions; all remaining positions set to zero. Neither actual activation values nor full support.',
              'selection': 'Five proteins selected by annotation count after preferred candidates absent; figure selects three by list index and commentary on apparent alignment, not independent validation.',
              'coordinates': 'Heatmap is zero-based. Annotation overlay converts one-based inclusive spans. Disulfide bond spans incorrectly fill interiors. Broad exceptions suppress missing overlays silently.',
              'annotation_scope': 'Reviewed annotations are not uniformly experimental. Site-count summary includes types not all plotted. Site-type count dictionary is initialized to zeros rather than accumulating counts.',
              'withdrawal': 'Localization, motif-specificity and across-family generalization claims withdrawn; aggregates cannot reconstruct full curves. No new pathogen-specific residue/feature analysis performed.',
              'replacement': 'ED10 now displays verified generic training-homolog-excluded probe controls, with uncertainty and counts from complete prediction exports.',
              'inputs': [file_identity(source, content=True)] + [file_identity(ROOT/p, content=True) for p in [
                  'scripts/scaled_1.5M/22_case_study_proteins.py',
                  'scripts/publication_figures/supplementary/ed10_case_studies.py',
                  'scripts/publication_figures/supplementary/sfig16_case_studies_full.py']],
              'script': file_identity(Path(__file__), content=True)}
    atomic_json(OUT/'case_study_plot_audit.json', report)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
