"""102. Variable-Length Causal Attention."""

import numpy as np


RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"tokens": 1, "width": 3, "sequences": 1},
    {"tokens": 13, "width": 5, "sequences": 4},
    {"tokens": 37, "width": 11, "sequences": 5},
]


def make_inputs(case, rng):
    tokens = case["tokens"]
    width = case["width"]
    sequences = case["sequences"]
    q = rng.standard_normal((tokens, width), dtype=np.float32)
    k = rng.standard_normal((tokens, width), dtype=np.float32)
    v = rng.standard_normal((tokens, width), dtype=np.float32)
    offsets = np.r_[
        0, np.sort(rng.choice(np.arange(1, tokens), sequences - 1, replace=False)), tokens
    ].astype(np.int32)
    return [q, k, v, offsets], q.shape, [np.uint32(tokens), np.uint32(width), np.uint32(sequences)]


def reference_impl(q, k, v, offsets, tokens, width, sequences):
    result = np.empty_like(q)
    for start, end in zip(offsets[:-1], offsets[1:]):
        scores = q[start:end] @ k[start:end].T / np.float32(np.sqrt(width))
        scores[np.triu_indices(end - start, 1)] = -np.inf
        weights = np.exp(scores - scores.max(1, keepdims=True))
        result[start:end] = (weights / weights.sum(1, keepdims=True)) @ v[start:end]
    return result


def dispatch_config(case):
    return case["tokens"] * case["width"], 1, 1, 64, 1, 1, False
