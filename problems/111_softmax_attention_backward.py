"""111. Softmax Attention Backward."""

import numpy as np


RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"queries": 1, "keys": 1, "width": 3},
    {"queries": 3, "keys": 5, "width": 7},
    {"queries": 13, "keys": 17, "width": 9},
]


def make_inputs(case, rng):
    queries = case["queries"]
    keys = case["keys"]
    width = case["width"]
    q = rng.standard_normal((queries, width), dtype=np.float32)
    k = rng.standard_normal((keys, width), dtype=np.float32)
    v = rng.standard_normal((keys, width), dtype=np.float32)
    grad = rng.standard_normal((queries, width), dtype=np.float32)
    return (
        [q, k, v, grad],
        [(q.shape, np.float32), (k.shape, np.float32), (v.shape, np.float32)],
        [np.uint32(queries), np.uint32(keys), np.uint32(width)],
    )


def reference_impl(q, k, v, grad, queries, keys, width):
    scale = 1 / np.sqrt(float(width))
    scores = q.astype(np.float64) @ k.T * scale
    probs = np.exp(scores - scores.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    dp = grad @ v.T
    ds = probs * (dp - (probs * dp).sum(1, keepdims=True)) * scale
    return (
        (ds @ k).astype(np.float32),
        (ds.T @ q).astype(np.float32),
        (probs.T @ grad).astype(np.float32),
    )


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
