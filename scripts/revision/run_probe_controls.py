"""Leakage-resistant probes and position/composition controls for Reviewer 2."""
import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from common import ROOT, RESULTS, OUT, SEED, summaries, labels, metrics, auc_bootstrap, full_column_shuffle, within_protein_shuffle
from probe_checkpoint import ProbeCheckpoint, atomic_json, file_identity


def legacy_shuffle(x, seed):
    rng = np.random.default_rng(seed)
    x = x.tocsc(copy=True)
    for j in range(x.shape[1]):
        a,b = x.indptr[j:j+2]
        x.indices[a:b] = rng.permutation(x.indices[a:b])
    x.has_sorted_indices = False
    x.sort_indices()
    return x.tocsr()


def fit_select(xs, ys, weights=None):
    candidates=[]
    for c in [.0001,.001,.01,.1]:
        clf=LogisticRegression(penalty='l1',C=c,solver='liblinear',max_iter=1000,
                               class_weight='balanced' if weights is None else None,random_state=SEED)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            clf.fit(xs[0],ys[0],sample_weight=weights)
        score=clf.decision_function(xs[1])
        candidates.append((roc_auc_score(ys[1],score),c,clf,
                           any(issubclass(w.category,ConvergenceWarning) for w in caught)))
    best=max(candidates,key=lambda z:z[0])
    return best[2],{'selected_C':best[1],'validation_auroc':float(best[0]),
                   'convergence_warning':best[3],
                   'selection_grid':{str(c):float(a) for a,c,_,_ in candidates}}


def retained_split_proteins(groups, split_groups, ids, excluded):
    keep = ~np.isin(ids, list(excluded))
    return [np.flatnonzero(np.isin(groups, g) & keep) for g in split_groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=OUT)
    parser.add_argument('--exclude-training-homologs', action='store_true')
    args = parser.parse_args()
    if args.exclude_training_homologs and args.run_dir.resolve() == OUT.resolve():
        parser.error('Exclusion refits require a separate --run-dir.')
    inputs = [ROOT/'data/eval_expanded/metadata.json', ROOT/'data/eval_expanded/sequences.json',
              OUT/'clusters.json']
    excluded = []
    if args.exclude_training_homologs:
        inputs.append(OUT/'training_homology_audit.json')
        excluded = json.loads(inputs[-1].read_text())['evaluation_ids_identity50']
    code = [Path(__file__), Path(__file__).with_name('common.py'),
            Path(__file__).with_name('probe_checkpoint.py')]
    identity = {'protocol': 1, 'seed': SEED, 'excluded_ids': sorted(excluded),
                'input_metadata': [file_identity(p, content=True) for p in inputs],
                'code': [file_identity(p, content=True) for p in code],
                'large_inputs': [file_identity(RESULTS/'sae_features'/m/'residue'/f)
                                 for m in ['esm3', 'esm2']
                                 for f in ['protein_summaries.h5', 'features_sparse.npz']]}
    checkpoint = ProbeCheckpoint(args.run_dir, identity)
    meta=json.loads((ROOT/'data/eval_expanded/metadata.json').read_text())
    seqs=json.loads((ROOT/'data/eval_expanded/sequences.json').read_text())
    cluster=json.loads((OUT/'clusters.json').read_text())
    ids,_,offsets=summaries('esm3')
    rng=np.random.default_rng(SEED)
    groups=np.array([cluster[p] for p in ids]); unique=np.unique(groups);rng.shuffle(unique)
    n=len(unique);split_groups=np.array_split(unique,[int(.6*n),int(.8*n)])
    split_proteins=retained_split_proteins(groups,split_groups,ids,excluded)
    row_to_protein=np.repeat(np.arange(len(ids)),offsets[:,1])
    assert np.array_equal(offsets[:,0],np.r_[0,np.cumsum(offsets[:-1,1])])
    rows=[]
    for ps,cap in zip(split_proteins,[200000,75000,150000]):
        ix=np.concatenate([np.arange(offsets[p,0],sum(offsets[p])) for p in ps])
        rows.append(np.sort(rng.choice(ix,min(cap,len(ix)),replace=False)))
    atomic_json(args.run_dir/'probe_split.json',{'seed':SEED,'train':[str(ids[p]) for p in split_proteins[0]],
         'validation':[str(ids[p]) for p in split_proteins[1]],'test':[str(ids[p]) for p in split_proteins[2]],
         'caps':[200000,75000,150000],'cluster_disjoint':True,
         'excluded_training_homologs':sorted(excluded),
         'split_rule':'Original cluster assignment retained; exclusions precede residue sampling and fitting.'})
    np.savez_compressed(args.run_dir/'probe_rows.npz',train=rows[0],validation=rows[1],test=rows[2])
    protein_groups=[row_to_protein[r] for r in rows]
    test_cluster=groups[protein_groups[2]]
    y=labels(ids,offsets,meta); ys=[y[r] for r in rows]
    output={'seed':SEED,'label_types':['Active site','Binding site','Metal binding','Site'],
        'negative_definition':'No selected annotation; not experimentally confirmed negative.',
        'split':'60/20/20 of detected 50%-identity, 80%-coverage sequence clusters',
        'sample_counts':{k:{'proteins':len(ps),'residues':len(r),'positive':int(v.sum()),'prevalence':float(v.mean())}
                         for k,ps,r,v in zip(['train','validation','test'],split_proteins,rows,ys)},
        'models':{}, 'excluded_training_homologs':sorted(excluded),
        'scope':'Fixed SAE dictionaries; all probes fitted after exclusions. Foundation-model training overlap is not assessed.'}
    output=checkpoint.load(output)
    expected={'y':ys[2],'protein':ids[protein_groups[2]],'cluster':test_cluster}
    for model in ['esm3','esm2']:
        mi,means,mo=summaries(model,'protein_mean_activations')
        assert np.array_equal(mi,ids) and np.array_equal(mo,offsets)
        print('Loading sparse',model,flush=True)
        full=sparse.load_npz(RESULTS/'sae_features'/model/'residue/features_sparse.npz')
        xs=[full[r].tocsr() for r in rows];del full
        output['models'].setdefault(model,{})
        def evaluate(name, xx, yy=ys, weight=None):
            if checkpoint.reusable(output,model,name,expected):
                print(model,name,'verified checkpoint; skipping',flush=True)
                return
            print(model,name,flush=True)
            if callable(xx):
                xx=xx()
            clf,selection=fit_select(xx,yy,weight)
            pred=clf.decision_function(xx[2])
            result={**metrics(ys[2],pred,protein_groups[2]),**selection,
                    'ci95_auroc':auc_bootstrap(ys[2],pred,test_cluster,n=1000)}
            output['models'][model][name]=result
            checkpoint.commit(output,model,name,{**expected,'score':pred},
                              {'coef':clf.coef_,'intercept':clf.intercept_,'classes':clf.classes_})
            print(result,flush=True)
        evaluate('intact',xs)
        binary=[x.copy() for x in xs]
        for x in binary:x.data[:]=1
        evaluate('binary_support',binary);del binary
        for seed in [2288,2289,2290]:
            evaluate(f'legacy_support_preserving_{seed}',lambda: [legacy_shuffle(x,seed+i*100) for i,x in enumerate(xs)])
            evaluate(f'global_column_shuffle_{seed}',lambda: [full_column_shuffle(x,seed+i*100) for i,x in enumerate(xs)])
            evaluate(f'within_protein_shuffle_{seed}',lambda: [within_protein_shuffle(x,g,seed+i*100) for i,(x,g) in enumerate(zip(xs,protein_groups))])
        # Same feature dimensionality, but identical predictions at every position in a protein.
        train_p=split_proteins[0]
        npos=np.bincount(protein_groups[0],weights=ys[0],minlength=len(ids))[train_p]
        ntotal=np.bincount(protein_groups[0],minlength=len(ids))[train_p]
        weights=np.r_[npos/(ys[0].mean()*2), (ntotal-npos)/((1-ys[0].mean())*2)]
        xx=[sparse.csr_matrix(np.concatenate([means[train_p],means[train_p]])),
            sparse.csr_matrix(means[protein_groups[1]]),sparse.csr_matrix(means[protein_groups[2]])]
        yy=[np.r_[np.ones(len(train_p)),np.zeros(len(train_p))],ys[1],ys[2]]
        evaluate('protein_mean_only',xx,yy,weights)
        del xx,means,xs
    aa='ACDEFGHIKLMNPQRSTVWY'; index={a:i for i,a in enumerate(aa)}
    composition=np.array([[seqs[p].count(a)/len(seqs[p]) for a in aa]+[np.log(len(seqs[p]))] for p in ids],dtype=np.float32)
    output['models'].setdefault('composition',{});model='composition'
    comp=[sparse.csr_matrix(composition[g]) for g in protein_groups]
    evaluate('protein_composition_length',comp)
    aa_flat=np.concatenate([np.array([index.get(a,20) for a in seqs[p]],dtype=np.int8) for p in ids])
    identity=[sparse.csr_matrix((np.ones(len(r)),(np.arange(len(r)),aa_flat[r])),shape=(len(r),21)) for r in rows]
    evaluate('residue_identity_plus_composition',[sparse.hstack([a,b],format='csr') for a,b in zip(identity,comp)])
    atomic_json(args.run_dir/'probe_controls.json',output)
    checkpoint.close()


if __name__=='__main__':
    main()
