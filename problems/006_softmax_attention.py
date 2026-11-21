"""6. Softmax Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"queries": 1, "keys": 1, "width": 3},
    {"queries": 7, "keys": 11, "width": 5},
    {"queries": 67, "keys": 43, "width": 16},
]


def make_inputs(case, rng):
    queries, keys, width = case["queries"], case["keys"], case["width"]
    q = rng.standard_normal((queries, width), dtype=np.float32)
    k = rng.standard_normal((keys, width), dtype=np.float32)
    v = rng.standard_normal((keys, width), dtype=np.float32)
    return [q, k, v], q.shape, [np.uint32(queries), np.uint32(keys), np.uint32(width)]


def reference_impl(q, k, v):
    scores = (q @ k.T) / np.float32(np.sqrt(q.shape[1]))
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return (weights / weights.sum(axis=1, keepdims=True)) @ v


def dispatch_config(case):
    return case["queries"] * case["width"], 1, 1, 64, 1, 1, False
