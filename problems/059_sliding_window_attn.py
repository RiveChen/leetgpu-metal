"""59. Sliding Window Self-Attention."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True

TEST_CASES = [
    {"length": 1, "width": 4, "window": 1},
    {"length": 7, "width": 5, "window": 8},
    {"length": 71, "width": 13, "window": 3},
]


def make_inputs(case, rng):
    length = case["length"]
    width = case["width"]
    window = case["window"]
    shape = (length, width)
    q, k, v = (rng.uniform(-1, 1, shape).astype(np.float32) for _ in range(3))
    return [q, k, v], shape, [np.uint32(length), np.uint32(width), np.uint32(window)]


def reference_impl(q, k, v, length, width, window):
    result = np.empty_like(q)
    for row in range(length):
        region = slice(max(0, row - int(window)), row + int(window) + 1)
        score = (k[region] @ q[row]) / np.float32(np.sqrt(width))
        weight = np.exp(score - score.max())
        result[row] = (weight / weight.sum()) @ v[region]
    return result


def dispatch_config(case):
    return case["length"] * case["width"], 1, 1, 64, 1, 1, False
