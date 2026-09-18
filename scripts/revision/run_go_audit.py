"""Complete feature-ID joins and GO reanalysis with explicit test families."""
import json
from collections import Counter
import numpy as np
from scipy import sparse, stats
from common import ROOT,RESULTS,OUT,SEED,save,summaries


def bh(p, family_size=None):
    n=len(p); m=n if family_size is None else family_size
    order=np.argsort(p); q=np.empty(n)
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*m/np.arange(1,n+1))[::-1])[::-1])
    return q


def main():
    meta=json.loads((ROOT/'data/eval_expanded/metadata.json').read_text())
    merged=json.loads((RESULTS/'decoder_pca.json').read_text())['feature_metadata']
    # The producer explicitly adds one to the zero-based encoder column at export.
    cats={r['feature_id']-1:r['cross_modal_category'] for r in merged}
    historical=json.loads((RESULTS/'go_enrichment.json').read_text())
    output={'feature_id_convention':'zero-based encoder/decoder column',
      'historical_problem':'GO export capped at first 500 enriched records; decoder_pca producer adds one to feature_id before joining zero-based GO records.',
      'study_rule':'At or above 90th percentile of nonzero per-protein maxima; >=5 nonzero proteins and >=3 study proteins.',
      'background':'All evaluated proteins with at least one GO annotation; direct annotations, no ancestor propagation.',
      'fdr':'BH within feature; revised family includes every background GO term; legacy candidate-only BH retained for reconciliation.',
      'models':{}}
    rng=np.random.default_rng(SEED)
    for model in ['esm3','esm2']:
        ids,x,_=summaries(model)
        ann=[{g['id'] for g in meta.get(p,{}).get('go_terms',[]) if g.get('id')} for p in ids]
        bg=np.array([bool(a) for a in ann]);x=x[bg];ann=[a for a in ann if a]
        terms=sorted(set.union(*ann)); ti={t:i for i,t in enumerate(terms)}
        rr=[];cc=[]
        for i,a in enumerate(ann):
            for t in a:rr.append(i);cc.append(ti[t])
        matrix=sparse.csr_matrix((np.ones(len(rr),dtype=np.int32),(rr,cc)),shape=(len(ann),len(terms)))
        pop=np.asarray(matrix.sum(0)).ravel();n=len(ann)
        old={r['feature_id']:r['n_enriched_terms'] for r in historical[model]['per_feature_results']}
        rows=[];mismatches=[]
        for fid in np.flatnonzero((x>0).any(0)):
            v=x[:,fid];nonzero=v>0;count=int(nonzero.sum())
            study=np.zeros(n,dtype=bool)
            if count>=5:study=v>=np.percentile(v[nonzero],90)
            ns=int(study.sum()); legacy=0; revised=0
            if ns>=3:
                overlap=np.asarray(matrix[study].sum(0)).ravel()
                present=overlap>0
                p=stats.hypergeom.sf(overlap[present]-1,n,pop[present],ns)
                legacy=int(np.sum((bh(p)<.05)&(overlap[present]>=2)))
                revised=int(np.sum((bh(p,len(terms))<.05)&(overlap[present]>=2)))
            if fid in old and old[fid]!=legacy:mismatches.append(int(fid))
            rows.append({'feature_id':int(fid),'category':cats.get(int(fid),'not_in_S_dictionary') if model=='esm3' else 'not_applicable',
                         'n_nonzero_proteins':count,'n_study_proteins':ns,
                         'n_terms_legacy':legacy,'n_terms_all_background_bh':revised})
            if len(rows)%1000==0:print(model,len(rows),'GO features',flush=True)
        n_active=historical[model]['n_active_features']
        row_map={r['feature_id']:r for r in rows}
        # Include active features with no activation on GO-annotated proteins as untestable zeros.
        _,all_x,_=summaries(model)
        for fid in np.flatnonzero((all_x>0).any(0)):
            if int(fid) not in row_map:
                rows.append({'feature_id':int(fid),'category':cats.get(int(fid),'not_in_S_dictionary') if model=='esm3' else 'not_applicable',
                    'n_nonzero_proteins':0,'n_study_proteins':0,'n_terms_legacy':0,'n_terms_all_background_bh':0})
        assert len(rows)==n_active
        del all_x
        result={'n_active':n_active,'n_background_proteins':n,'n_background_terms':len(terms),
                'legacy_stored_record_mismatches':mismatches,'n_historical_records':len(old),
                'n_enriched_legacy':sum(r['n_terms_legacy']>0 for r in rows),
                'n_enriched_revised':sum(r['n_terms_all_background_bh']>0 for r in rows),
                'per_feature':rows,'categories':{}}
        for cat in sorted({r['category'] for r in rows}):
            cr=[r for r in rows if r['category']==cat]
            v=np.array([r['n_terms_all_background_bh'] for r in cr]);nz=v[v>0]
            bootstrap=[np.mean(rng.choice(v,len(v),replace=True)>0) for _ in range(2000)]
            result['categories'][cat]={'n_features':len(v),'n_testable':sum(r['n_study_proteins']>=3 for r in cr),
                'n_enriched':int((v>0).sum()),'frac_enriched':float(np.mean(v>0)),
                'median_all_features':float(np.median(v)),'median_enriched_only':float(np.median(nz)) if len(nz) else None,
                'feature_bootstrap_fraction_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
                'ci_scope':'descriptive resampling of dictionary features; not independent biological replicates'}
        if model=='esm3':
            enh=[r for r in rows if r['category']=='enhanced']; inv=[r for r in rows if r['category']=='invariant']
            a=np.array([r['n_terms_all_background_bh'] for r in enh]);b=np.array([r['n_terms_all_background_bh'] for r in inv])
            delta=[np.median(rng.choice(a,len(a),replace=True))-np.median(rng.choice(b,len(b),replace=True)) for _ in range(2000)]
            result['enhanced_minus_invariant_median_ci95']=np.quantile(delta,[.025,.975]).tolist()
            # Match each enhanced feature to closest study-size invariant (without replacement).
            remaining=inv.copy();paired=[]
            for r in sorted(enh,key=lambda r:r['n_study_proteins']):
                j=min(range(len(remaining)),key=lambda j:abs(np.log1p(remaining[j]['n_study_proteins'])-np.log1p(r['n_study_proteins'])))
                control=remaining.pop(j)
                paired.append((r['n_terms_all_background_bh'],control['n_terms_all_background_bh']))
            paired=np.array(paired)
            result['study_size_matched']={'n_pairs':len(paired),'median_enhanced':float(np.median(paired[:,0])),
              'median_invariant':float(np.median(paired[:,1])),'mean_paired_difference':float(np.mean(paired[:,0]-paired[:,1])),
              'caveat':'Exploratory greedy study-size matching; does not account for all feature dependence or GO hierarchy.'}
        output['models'][model]=result
        save('go_audit.json',output)
        print(model,result['categories'],flush=True)
    save('go_audit.json',output)


if __name__=='__main__':main()
