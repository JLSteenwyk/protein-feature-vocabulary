"""Portable provenance validation must fail closed without mutating the archive."""
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/revision'))
from verify_relocated_manifest import records, relocate, verify_row


def test_relocation_requires_explicit_component_prefixes(tmp_path):
    mappings = [(Path('/original/project'), tmp_path)]
    assert relocate('/original/project/a.bin', tmp_path, mappings) == tmp_path/'a.bin'
    with pytest.raises(ValueError, match='explicit prefix'):
        relocate('/original/project-extra/a.bin', tmp_path, mappings)
    with pytest.raises(ValueError, match='Parent traversal'):
        relocate('../a.bin', tmp_path, mappings)


def test_relocated_hashes_ignore_mtime_but_not_content(tmp_path):
    path = tmp_path/'sample.bin'
    path.write_bytes(b'abc')
    row = {'path': '/old/sample.bin', 'bytes': 3, 'mtime_ns': 1,
           'sha256': hashlib.sha256(b'abc').hexdigest()}
    before = dict(row)
    assert verify_row(row, tmp_path, [(Path('/old'), tmp_path)])['bytes'] == 3
    assert row == before
    path.write_bytes(b'abd')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        verify_row(row, tmp_path, [(Path('/old'), tmp_path)])
    path.write_bytes(b'abcd')
    with pytest.raises(ValueError, match='Byte-count mismatch'):
        verify_row(row, tmp_path, [(Path('/old'), tmp_path)])


def test_supported_manifest_schemas_and_longest_mapping(tmp_path):
    row = {'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()}
    inventory = {'schema': 1, 'path_base': 'project root', 'files': {'a': row}}
    assert list(records(inventory))[0][1] == dict(row, path='a')
    assert len(list(records({'nested': [dict(row, path='a')]}))) == 1
    mappings = [(Path('/old'), tmp_path/'outer'), (Path('/old/models'), tmp_path/'models')]
    assert relocate('/old/models/a', tmp_path, mappings) == tmp_path/'models/a'
