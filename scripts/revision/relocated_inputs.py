"""Explicit in-memory input relocation; archived identity JSON remains immutable."""
from copy import deepcopy
from pathlib import Path

from verify_relocated_manifest import relocate


def parse_mappings(values):
    mappings = []
    for value in values:
        old, separator, new = value.partition('=')
        if not separator or not Path(old).is_absolute() or not Path(new).is_absolute():
            raise ValueError('Mapping requires absolute OLD=NEW prefixes')
        if '..' in Path(old).parts or '..' in Path(new).parts or any(Path(old) == pair[0] for pair in mappings):
            raise ValueError('Duplicate or noncanonical mapping prefix')
        mappings.append((Path(old), Path(new)))
    return mappings


def mapped_identity(identity, mappings, root):
    """Map only content-identified file paths; caller still verifies every hash."""
    result = deepcopy(identity)
    if not mappings:
        return result

    def visit(value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                value['path'] = str(relocate(value['path'], root, mappings))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(result)
    return result
