"""Decode generic ESM SS8 outputs without treating special tokens as SS3 states."""
import numpy as np
from scipy.special import logsumexp

SS8_VOCAB = ('<pad>', '<motif>', '<unk>', 'G', 'H', 'I', 'T', 'E', 'B', 'S', 'C')
SS3_STATES = ('H', 'E', 'C')
SS8_TO_SS3 = {'G': 'H', 'H': 'H', 'I': 'H', 'E': 'E', 'B': 'E', 'T': 'C', 'S': 'C', 'C': 'C'}


def ss3_predictions(logits, vocabulary=SS8_VOCAB):
    """Argmax after summing valid SS8 probability mass into H/E/C classes.

    Returns integer H/E/C indices. Input is per-residue logits, with special
    sequence positions already excluded by the caller. This is prediction
    agreement infrastructure, not an accuracy measurement against ground truth.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim < 1 or logits.shape[-1] != len(vocabulary):
        raise ValueError('Logit width must equal the supplied SS8 vocabulary width')
    if len(set(vocabulary)) != len(vocabulary) or not set(SS8_TO_SS3) <= set(vocabulary):
        raise ValueError('Vocabulary must contain each biological SS8 state exactly once')
    if not np.isfinite(logits).all():
        raise ValueError('Secondary-structure logits must be finite')
    # logsumexp is proportional to grouped probability mass; the normalizer cancels.
    grouped = [logsumexp(logits[..., [i for i, token in enumerate(vocabulary)
                                     if SS8_TO_SS3.get(token) == state]], axis=-1)
               for state in SS3_STATES]
    return np.stack(grouped, axis=-1).argmax(axis=-1)
