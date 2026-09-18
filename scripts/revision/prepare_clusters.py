"""Create explicit sequence-identity clusters for revision data splits."""
import json
import subprocess
from common import ROOT, OUT, save, summaries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids, _, offsets = summaries('esm3')
    seqs = json.loads((ROOT/'data/eval_expanded/sequences.json').read_text())
    assert len(set(ids)) == len(ids)
    fasta = OUT/'evaluation.fasta'
    with fasta.open('w') as f:
        for pid, (_, length) in zip(ids, offsets):
            assert pid in seqs and len(seqs[pid]) == length, pid
            f.write(f'>{pid}\n{seqs[pid]}\n')
    command = ['mmseqs','easy-cluster',str(fasta),str(OUT/'identity50'),
               str(OUT/'mmseqs_tmp'),'--min-seq-id','0.5','-c','0.8',
               '--cov-mode','0','--cluster-mode','1','--threads','16']
    with (OUT/'mmseqs.log').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    clusters = {}
    for line in (OUT/'identity50_cluster.tsv').read_text().splitlines():
        representative, member = line.split('\t')
        clusters[member] = representative
    assert set(clusters) == set(ids)
    save('clusters.json',clusters)
    save('cluster_provenance.json', {'command':command,'version':subprocess.check_output(['mmseqs','version'],text=True).strip(),
         'n_proteins':len(ids),'n_clusters':len(set(clusters.values())),
         'scope':'Evaluation-set clustering only; not a new audit of SAE training overlap.',
         'caveat':'Connected components of detected similarity graph; search sensitivity limits detection.'})
    print('Clusters',len(set(clusters.values())),flush=True)


if __name__ == '__main__':
    main()
