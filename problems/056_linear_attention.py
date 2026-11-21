"""56. Linear Self-Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True

TEST_CASES = [{"length": 1, "width": 3}, {"length": 9, "width": 8}, {"length": 131, "width": 17}]


def make_inputs(case, rng):
    length = case["length"]
    width = case["width"]
    shape = (length, width)
    q, k, v = (rng.uniform(-1, 1, shape).astype(np.float32) for _ in range(3))
    return [q, k, v], shape, [np.uint32(length), np.uint32(width)]


def reference_impl(q, k, v, length, width):
    pq = np.where(q > 0, q + 1, np.exp(q))
    pk = np.where(k > 0, k + 1, np.exp(k))
    return (pq @ (pk.T @ v)) / (pq @ pk.sum(axis=0))[:, None]


def dispatch_config(case):
    return case["width"], 1, 1, 128, 1, 1, True
