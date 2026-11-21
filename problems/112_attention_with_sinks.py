"""112. Attention with Sinks."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"length": 1, "width": 1, "sinks": 1, "window": 1},
    {"length": 2, "width": 7, "sinks": 1, "window": 1},
    {"length": 39, "width": 9, "sinks": 3, "window": 5},
]


def make_inputs(case, rng):
    length = case["length"]
    width = case["width"]
    sinks = case["sinks"]
    window = case["window"]
    q = rng.uniform(-1, 1, (length, width)).astype(np.float32)
    k = rng.uniform(-1, 1, (length, width)).astype(np.float32)
    v = rng.uniform(-1, 1, (length, width)).astype(np.float32)
    return (
        [q, k, v],
        q.shape,
        [np.uint32(length), np.uint32(width), np.uint32(sinks), np.uint32(window)],
    )


def reference_impl(q, k, v, length, width, sinks, window):
    rows = np.arange(length)[:, None]
    cols = np.arange(length)[None, :]
    allowed = (cols <= rows) & ((cols < sinks) | (cols + int(window) > rows))
    scores = q @ k.T / np.float32(np.sqrt(width))
    scores[~allowed] = -np.inf
    weights = np.exp(scores - scores.max(1, keepdims=True))
    return (weights / weights.sum(1, keepdims=True)) @ v


def dispatch_config(case):
    return case["length"] * case["width"], 1, 1, 64, 1, 1, False
