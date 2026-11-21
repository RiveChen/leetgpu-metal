"""53. Causal Self-Attention."""

import numpy as np

RTOL, ATOL = 1e-4, 1e-4
TEST_CASES = [{"m": m, "width": d} for m, d in ((1, 5), (11, 3), (47, 16))]


def make_inputs(case, rng):
    m, width = case["m"], case["width"]
    q, k, v = (rng.standard_normal((m, width), dtype=np.float32) for _ in range(3))
    return [q, k, v], q.shape, [np.uint32(m), np.uint32(width)]


def reference_impl(q, k, v):
    scores = (q @ k.T) / np.sqrt(q.shape[-1])
    scores[np.triu_indices(len(q), k=1)] = -np.inf
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return ((weights / weights.sum(axis=-1, keepdims=True)) @ v).astype(np.float32)


def dispatch_config(case):
    return case["m"], 1, 1, 64, 1, 1, False
