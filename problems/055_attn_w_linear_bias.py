"""55. Attention with Linear Biases."""

import numpy as np

REFERENCE_USES_SCALARS = True
RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"queries": 1, "keys": 3, "width": 4, "alpha": -0.625},
    {"queries": 11, "keys": 7, "width": 5, "alpha": 0.375},
    {"queries": 33, "keys": 51, "width": 13, "alpha": 0},
]


def make_inputs(case, rng):
    queries, keys, width, alpha = case["queries"], case["keys"], case["width"], case["alpha"]
    q = rng.standard_normal((queries, width), dtype=np.float32)
    k = rng.standard_normal((keys, width), dtype=np.float32)
    v = rng.standard_normal((keys, width), dtype=np.float32)
    return [q, k, v], q.shape, [np.uint32(n) for n in (queries, keys, width)] + [np.float32(alpha)]


def reference_impl(q, k, v, queries, keys, width, alpha):
    bias = alpha * (np.arange(queries)[:, None] - np.arange(keys)[None, :])
    scores = (q @ k.T) / np.sqrt(float(width)) + bias
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return ((weights / weights.sum(axis=1, keepdims=True)) @ v).astype(np.float32)


def dispatch_config(case):
    return case["queries"] * case["width"], 1, 1, 64, 1, 1, False
