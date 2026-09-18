"""Capture local dependency versions and streaming checksums of fixed revision inputs."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from common import ROOT, OUT, RESULTS
from probe_checkpoint import atomic_json


def stable_digest(path, chunk_size=8*1024*1024):
    path = Path(path)
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(chunk_size), b''):
            checksum.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f'File changed during checksum: {path}')
    return {'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns, 'sha256': checksum.hexdigest()}


def version_command(command):
    executable = shutil.which(command[0])
    if executable is None:
        return {'available': False}
    result = subprocess.run([executable, *command[1:]], text=True, capture_output=True, timeout=30)
    return {'available': True, 'executable': executable, 'returncode': result.returncode,
            'output': (result.stdout + result.stderr).strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rehash', action='store_true', help='Recompute even existing content digests.')
    args = parser.parse_args()
    directory = ROOT/'revision/reproducibility'
    directory.mkdir(exist_ok=True)
    lock = (directory/'.capture.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    packages = sorted({(d.metadata['Name'], d.version) for d in importlib.metadata.distributions()
                       if d.metadata['Name']}, key=lambda x: x[0].lower())
    software = {'captured_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'python_executable': sys.executable, 'platform': platform.platform(),
                'machine': platform.machine(), 'packages': dict(packages),
                'native_tools': {name: version_command(command) for name, command in {
                    'mmseqs2': ['mmseqs', 'version'], 'pandoc': ['pandoc', '--version'],
                    'pdftotext': ['pdftotext', '-v'],
                    'tectonic': ['/tmp/bioinf-tectonic/tectonic', '--version']}.items()},
                'scope': 'Observed working environment, not a portable installation guarantee. No environment variables or credentials collected.'}
    atomic_json(directory/'software_inventory.json', software)
    freeze = directory/'requirements-observed.txt'
    temporary = freeze.with_suffix('.tmp')
    temporary.write_text('# Observed versions, not a tested cross-platform installation lock.\n' +
                         ''.join(f'{name}=={version}\n' for name, version in packages))
    temporary.replace(freeze)

    paths = [ROOT/'INTERPRETABILITY.zip', ROOT/'data/eval_expanded/metadata.json',
             ROOT/'data/eval_expanded/sequences.json', ROOT/'data/sae_training/uniref50_1.5M.fasta',
             OUT/'clusters.json', OUT/'identity50_cluster.tsv', OUT/'evaluation.fasta',
             OUT/'training_homology_audit.json', OUT/'evaluation_vs_training.tsv']
    for model in ['esm3', 'esm2']:
        paths += [ROOT/'models/sae_1.5M'/f'{model}_residue_ef8_k64/best.pt']
        paths += [RESULTS/'sae_features'/model/'residue'/name
                  for name in ['features_sparse.npz', 'protein_summaries.h5']]
    for condition in ['S_only', 'S_St']:
        chunks = sorted((RESULTS/'cross_modal'/condition/'residue_L33').glob('*.h5'))
        if not chunks:
            raise FileNotFoundError(f'No cross-modal input chunks for {condition}')
        paths += chunks
    paths += sorted((ROOT/'revision/submitted').glob('*'))
    paths = sorted({p for p in paths if not p.is_dir()})
    destination = directory/'input_manifest.json'
    manifest = json.loads(destination.read_text()) if destination.exists() else {
        'schema': 1, 'algorithm': 'sha256', 'path_base': 'project root', 'files': {}}
    manifest.update({'expected_files': len(paths), 'complete': False,
                     'scope': 'Fixed inputs for corrected statistical/probe analyses, paired SAE re-encoding and original submission. '
                              'Does not checksum every historical training-activation chunk or foundation-model cache. '
                              'Mutable analysis outputs and source files require a separate final snapshot after active jobs finish.'})
    atomic_json(destination, manifest)
    for path in paths:
        name = str(path.relative_to(ROOT))
        stat = path.stat()
        old = manifest['files'].get(name)
        if not args.rehash and old and (old['bytes'], old['mtime_ns']) == (stat.st_size, stat.st_mtime_ns):
            print('Cached', name, flush=True)
            continue
        manifest['files'][name] = stable_digest(path)
        atomic_json(destination, manifest)
        print('Hashed', name, flush=True)
    expected = {str(p.relative_to(ROOT)) for p in paths}
    manifest['files'] = {name: value for name, value in manifest['files'].items() if name in expected}
    manifest['complete'] = len(manifest['files']) == len(paths)
    manifest['captured_utc'] = datetime.now(timezone.utc).isoformat()
    atomic_json(destination, manifest)
    print('Complete:', len(manifest['files']), 'fixed input hashes', flush=True)
    lock.close()


if __name__ == '__main__':
    main()
