"""Generic fixed-direction representation controls, not biological optimization."""
from contextlib import contextmanager
import numpy as np
import torch


STRENGTHS = (-50., -20., -10., -5., 0., 5., 10., 20., 50.)
SEEDS = (2288, 2289, 2290, 2291, 2292)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid direction')
    return vector/norm


def make_directions(weights, target_ids, categories, seeds=SEEDS):
    weights = np.asarray(weights, dtype=np.float32)
    target_ids = list(target_ids)
    if (weights.ndim != 2 or len(target_ids) != 20 or len(set(target_ids)) != 20
            or min(target_ids) < 0 or max(target_ids) >= weights.shape[1]
            or not np.isfinite(weights).all()
            or set(categories) != set(range(weights.shape[1]))):
        raise ValueError('Invalid decoder, target IDs or complete categories')
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('Distinct control seeds required')
    target = unit(weights[:, target_ids].mean(axis=1))
    pool = [i for i in range(weights.shape[1]) if categories[i] == 'unclassified' and i not in target_ids]
    if len(pool) < 20:
        raise ValueError('Insufficient unclassified decoder controls')
    vectors, metadata = {'target': target}, {'target': {'feature_ids': target_ids, 'kind': 'target'}}
    for seed in seeds:
        # Independent streams: the orthogonal control is not the random control projected.
        streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        random = unit(streams[0].normal(size=len(target)))
        orth = streams[1].normal(size=len(target)).astype(np.float32)
        orth = unit(orth-np.dot(orth, target)/np.dot(target, target)*target)
        ids = streams[2].choice(pool, 20, replace=False).tolist()
        for kind, vector in [('random', random), ('orthogonal', orth),
                             ('decoder', unit(weights[:, ids].mean(axis=1)))]:
            name = f'{kind}_{seed}'
            vectors[name] = vector
            metadata[name] = {'kind': kind, 'seed': seed,
                              'feature_ids': ids if kind == 'decoder' else [],
                              'target_cosine': float(np.dot(vector, target))}
    return vectors, metadata


def condition_order(directions):
    return [('normal', None, 0.)] + [
        (f'{name}_alpha_{alpha:g}', name, alpha)
        for name in directions for alpha in STRENGTHS] + [('restored', None, 0.)]


def add_direction(hidden, direction, alpha):
    if hidden.ndim != 3 or not hidden.is_floating_point() or not torch.isfinite(hidden).all():
        raise ValueError('Invalid hidden tensor')
    vector = torch.as_tensor(direction, device=hidden.device, dtype=torch.float32)
    if vector.shape != (hidden.shape[-1],) or not torch.isfinite(vector).all() or not np.isfinite(alpha):
        raise ValueError('Invalid direction shape or strength')
    if alpha == 0:
        return hidden
    # Preserve the historical float32 addition, including all special-token positions.
    result = hidden + vector*float(alpha)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite perturbed state')
    return result


@contextmanager
def steering_intervention(block, direction, alpha):
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        changed = add_direction(hidden, direction, alpha)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed
    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
