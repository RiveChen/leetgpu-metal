"""88. Shared-Prefix Cascade Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001

TEST_CASES = [
    {"batch": 1, "heads": 1, "prefix": 1, "suffix": 1, "width": 3},
    {"batch": 3, "heads": 2, "prefix": 5, "suffix": 7, "width": 8},
    {"batch": 5, "heads": 4, "prefix": 13, "suffix": 9, "width": 7},
]


def make_inputs(case, rng):
    batch, heads, prefix, suffix, width = (
        case["batch"],
        case["heads"],
        case["prefix"],
        case["suffix"],
        case["width"],
    )
    q = rng.standard_normal((batch, heads, suffix, width), dtype=np.float32)
    shared_k = rng.standard_normal((heads, prefix, width), dtype=np.float32)
    shared_v = rng.standard_normal(shared_k.shape, dtype=np.float32)
    local_k = rng.standard_normal(q.shape, dtype=np.float32)
    local_v = rng.standard_normal(q.shape, dtype=np.float32)
    return (
        [q, shared_k, shared_v, local_k, local_v],
        q.shape,
        [np.uint32(v) for v in (batch, heads, prefix, suffix, width)],
    )


def reference_impl(q, shared_k, shared_v, local_k, local_v):
    out = np.empty_like(q)
    for b in range(len(q)):
        keys = np.concatenate((shared_k, local_k[b]), axis=1)
        values = np.concatenate((shared_v, local_v[b]), axis=1)
        scores = q[b] @ keys.swapaxes(1, 2) / np.float32(np.sqrt(q.shape[-1]))
        future = np.arange(keys.shape[1]) > shared_k.shape[1] + np.arange(q.shape[2])[:, None]
        scores = np.where(future, -np.inf, scores)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        out[b] = (weights / weights.sum(axis=-1, keepdims=True)) @ values
    return out


def dispatch_config(case):
    return case["batch"] * case["heads"] * case["suffix"] * case["width"], 1, 1, 64, 1, 1, False
