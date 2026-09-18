"""Export the complete GO test family and study membership, not selected hits."""
import json
import numpy as np
from scipy import sparse, stats
from common import ROOT, OUT, summaries
from probe_checkpoint import atomic_json, file_identity
from run_go_audit import bh


def complete_term_test(overlap, population, n_background, n_study, eligible):
    overlap = np.asarray(overlap)
    p = np.ones(len(population), dtype=float)
    if eligible:
        p = stats.hypergeom.sf(overlap - 1, n_background, population, n_study)
    q = bh(p)
    enriched = eligible & (q < .05) & (overlap >= 2)
    return p, q, enriched


def main():
    directory = OUT / 'go_complete_tests'
    directory.mkdir(exist_ok=True)
    metadata_path = ROOT / 'data/eval_expanded/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    audit = json.loads((OUT / 'go_audit.json').read_text())
    manifest = {'schema': 1, 'feature_ids': 'zero-based encoder columns',
                'p_q_axes': 'feature_ids by term_ids; includes all background terms and active features',
                'ineligible_convention': 'p=q=1, enriched=false; consult eligible mask',
                'study_rule': audit['study_rule'], 'background': audit['background'],
                'fdr': 'BH over all background terms within each eligible feature',
                'metadata': file_identity(metadata_path, content=True), 'models': {}}
    for model in ['esm3', 'esm2']:
        ids, x, _ = summaries(model)
        active = np.flatnonzero((x > 0).any(0))
        annotations = [{g['id'] for g in metadata.get(pid, {}).get('go_terms', []) if g.get('id')}
                       for pid in ids]
        keep = np.array([bool(a) for a in annotations])
        ids, x = ids[keep], x[keep]
        annotations = [a for a in annotations if a]
        terms = sorted(set.union(*annotations))
        index = {term: i for i, term in enumerate(terms)}
        rows, columns = [], []
        for row, ann in enumerate(annotations):
            for term in sorted(ann):
                rows.append(row); columns.append(index[term])
        membership = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, columns)),
                                       shape=(len(ids), len(terms)))
        population = np.asarray(membership.sum(0)).ravel()
        pvalues = np.ones((len(active), len(terms)), dtype=float)
        qvalues = np.ones_like(pvalues)
        overlaps = np.zeros(pvalues.shape, dtype=np.int32)
        eligible = np.zeros(len(active), dtype=bool)
        study_rows, study_columns = [], []
        counts = np.zeros(len(active), dtype=int)
        expected = {r['feature_id']: r for r in audit['models'][model]['per_feature']}
        for row, fid in enumerate(active):
            value = x[:, fid]
            nonzero = value > 0
            study = np.zeros(len(ids), dtype=bool)
            if nonzero.sum() >= 5:
                study = value >= np.percentile(value[nonzero], 90)
            ns = int(study.sum())
            eligible[row] = ns >= 3
            overlap = np.asarray(membership[study].sum(0)).ravel()
            pvalues[row], qvalues[row], enriched = complete_term_test(
                overlap, population, len(ids), ns, eligible[row])
            overlaps[row] = overlap
            counts[row] = enriched.sum()
            if counts[row] != expected[int(fid)]['n_terms_all_background_bh']:
                raise ValueError(f'Complete export disagrees with audit for {model} feature {fid}')
            if ns != expected[int(fid)]['n_study_proteins']:
                raise ValueError(f'Study membership disagrees for {model} feature {fid}')
            study_rows.extend([row] * ns)
            study_columns.extend(np.flatnonzero(study).tolist())
            if (row + 1) % 1000 == 0:
                print(model, row + 1, 'complete term families verified', flush=True)
        studies = sparse.csr_matrix((np.ones(len(study_rows), dtype=np.int8), (study_rows, study_columns)),
                                    shape=(len(active), len(ids)))
        target = directory / f'{model}_tests.npz'
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, feature_ids=active, term_ids=np.array(terms),
                            background_protein_ids=ids, population=population,
                            eligible=eligible, p=pvalues, q=qvalues, overlap=overlaps,
                            n_enriched=counts, study_indptr=studies.indptr,
                            study_indices=studies.indices, study_data=studies.data,
                            study_shape=studies.shape, annotation_indptr=membership.indptr,
                            annotation_indices=membership.indices, annotation_data=membership.data,
                            annotation_shape=membership.shape)
        temporary.replace(target)
        manifest['models'][model] = {'file': target.name, 'n_features': len(active),
                                     'n_terms': len(terms), 'n_background_proteins': len(ids),
                                     'n_eligible': int(eligible.sum()), 'count_mismatches': 0,
                                     'export': file_identity(target, content=True)}
        atomic_json(directory / 'manifest.json', manifest)
        print(model, 'export complete', flush=True)


if __name__ == '__main__':
    main()
