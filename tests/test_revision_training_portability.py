"""Run the real training auditor against small relocated archives."""
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/revision'))
from run_training_provenance import run
import audit_training_provenance as audit


def fixture(tmp_path):
    storage, project, analysis = [tmp_path/name for name in ['storage', 'project', 'analysis']]
    analysis.mkdir()
    (analysis/'evaluation_vs_training.tsv').write_text('query\ttrain0\t70\n')
    for model in ['esm2', 'esm3']:
        directory = storage/model/'residue_L33'
        directory.mkdir(parents=True)
        with h5py.File(directory/'chunk.h5', 'w') as f:
            f.create_dataset('activations', shape=(2000, 2), dtype='f4')
            f.create_dataset('ids', data=np.array([b'train0', b'train1']))
            f.create_dataset('offsets', data=np.array([[0, 0, 1000], [1, 1000, 1000]]))
        checkpoint = project/f'models/sae_1.5M/{model}_residue_ef8_k64/best.pt'
        checkpoint.parent.mkdir(parents=True)
        torch.save({'train_config': {'batch_size': 100, 'num_epochs': 5}, 'step': 50}, checkpoint)
    return storage, project, analysis


def test_real_auditor_relocates_without_overwriting(tmp_path):
    storage, project, analysis = fixture(tmp_path)
    archive = analysis/'training_provenance_audit.json'
    archive.write_text('historical result')
    original = audit.STORAGE, audit.ROOT, audit.OUT, audit.save
    output = tmp_path/'new-audit.json'
    result = run(storage, project, analysis, output)
    assert result['complete']
    assert (audit.STORAGE, audit.ROOT, audit.OUT, audit.save) == original
    assert archive.read_text() == 'historical result'
    for model in ['esm2', 'esm3']:
        row = result['audit']['models'][model]
        assert row['sampled_residues'] == 2000
        assert row['reconstructed_training_residues'] == 1000
        assert row['checkpoint_step_matches_current_sampler']
        assert row['evaluation_ids_homologous_to_reconstructed_training'] == ['query']
        assert not row['historical_selection_verified']
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        run(storage, project, analysis, output)
    assert output.read_bytes() == before


def test_auditor_failure_restores_globals_and_records_incomplete(tmp_path):
    storage, project, analysis = fixture(tmp_path)
    original = audit.STORAGE, audit.ROOT, audit.OUT, audit.save
    with h5py.File(storage/'esm3/residue_L33/chunk.h5', 'a') as f:
        del f['offsets']
    output = tmp_path/'failed.json'
    with pytest.raises(KeyError):
        run(storage, project, analysis, output)
    report = json.loads(output.read_text())
    assert report['status'] == 'failed' and not report['complete']
    assert (audit.STORAGE, audit.ROOT, audit.OUT, audit.save) == original
