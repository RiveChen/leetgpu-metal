"""80. Grouped Query Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001

TEST_CASES = [
    {"heads": 2, "kv_heads": 1, "length": 1, "width": 4},
    {"heads": 4, "kv_heads": 4, "length": 7, "width": 5},
    {"heads": 6, "kv_heads": 2, "length": 19, "width": 12},
]


def make_inputs(case, rng):
    heads, kv_heads, length, width = case["heads"], case["kv_heads"], case["length"], case["width"]
    q = rng.standard_normal((heads, length, width), dtype=np.float32)
    k = rng.standard_normal((kv_heads, length, width), dtype=np.float32)
    v = rng.standard_normal(k.shape, dtype=np.float32)
    return (
        [q, k, v],
        q.shape,
        [np.uint32(v) for v in (heads, kv_heads, length, width)],
    )


def reference_impl(q, k, v):
    repeats = q.shape[0] // k.shape[0]
    keys = np.repeat(k, repeats, axis=0)
    values = np.repeat(v, repeats, axis=0)
    scores = q @ keys.swapaxes(-1, -2) / np.float32(np.sqrt(q.shape[-1]))
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (weights / weights.sum(axis=-1, keepdims=True)) @ values


def dispatch_config(case):
    return case["heads"] * case["length"] * case["width"], 1, 1, 64, 1, 1, False
