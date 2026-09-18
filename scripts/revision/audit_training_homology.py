"""Search evaluation sequences against archived SAE-training candidates."""
import hashlib
import shutil
import subprocess
from common import ROOT, OUT, save


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    executable = shutil.which('mmseqs')
    if executable is None:
        raise RuntimeError('MMseqs2 is required; no fallback to length folds')
    query = OUT / 'evaluation.fasta'
    target = ROOT / 'data/sae_training/uniref50_1.5M.fasta'
    result = OUT / 'evaluation_vs_training.tsv'
    command = [executable, 'easy-search', str(query), str(target), str(result),
               str(OUT / 'training_homology_tmp'), '--min-seq-id', '0.5',
               '-s', '7.5', '-e', '1e-5', '--threads', '8', '--max-seqs', '300',
               '--format-output', 'query,target,pident,alnlen,qcov,tcov,evalue']
    provenance = {'command': command, 'query_sha256': digest(query), 'target_sha256': digest(target),
                  'version': subprocess.check_output([executable, 'version'], text=True).strip()}
    save('training_homology_provenance.json', provenance)
    with (OUT / 'training_homology.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    any_hit, coverage_hit = set(), set()
    n = 0
    with result.open() as handle:
        for line in handle:
            query_id, _, identity, _, qcov, tcov, _ = line.rstrip().split('\t')
            if float(identity) < 50:
                continue
            n += 1
            any_hit.add(query_id)
            if float(qcov) >= .8 and float(tcov) >= .8:
                coverage_hit.add(query_id)
    save('training_homology_audit.json', {
        'n_retained_hit_rows': n, 'n_evaluation_with_hit_identity50': len(any_hit),
        'n_evaluation_with_hit_identity50_coverage80': len(coverage_hit),
        'evaluation_ids_identity50': sorted(any_hit),
        'evaluation_ids_identity50_coverage80': sorted(coverage_hit),
        'scope': 'Detected local alignment hits against archived candidate training sequences; '
                 'not all-pairs proof, not verification of successful SAE training extraction, '
                 'and not an audit of foundation-model pretraining. No coverage threshold in search; '
                 '80% bidirectional coverage is a secondary filter on reported hits.'})
    print('Training-homology search complete:', len(any_hit), 'with hits;', len(coverage_hit), 'at coverage >=80%', flush=True)


if __name__ == '__main__':
    main()
