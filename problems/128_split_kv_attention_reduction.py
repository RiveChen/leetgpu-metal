"""128. Split-KV Attention Reduction."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"splits": 1, "heads": 3, "width": 2},
    {"splits": 5, "heads": 2, "width": 7},
    {"splits": 19, "heads": 13, "width": 17},
]


def make_inputs(case, rng):
    splits = case["splits"]
    heads = case["heads"]
    width = case["width"]
    parts = rng.standard_normal((splits, heads, width), dtype=np.float32)
    lse = rng.uniform(-110, 110, (splits, heads)).astype(np.float32)
    return [parts, lse], (heads, width), [np.uint32(v) for v in parts.shape]


def reference_impl(parts, lse):
    weights = np.exp(lse - lse.max(axis=0))
    return np.einsum("sh,shd->hd", weights / weights.sum(axis=0), parts)


def dispatch_config(case):
    return case["heads"] * case["width"], 1, 1, 64, 1, 1, False
