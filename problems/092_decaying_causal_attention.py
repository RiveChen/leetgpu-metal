"""92. Decaying Causal Attention."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"length": 1, "width": 3, "decay": 0.5},
    {"length": 7, "width": 5, "decay": 1.0},
    {"length": 37, "width": 11, "decay": 0.83},
]


def make_inputs(case, rng):
    length = case["length"]
    width = case["width"]
    decay = case["decay"]
    q = rng.standard_normal((length, width), dtype=np.float32)
    k = rng.standard_normal((length, width), dtype=np.float32)
    v = rng.standard_normal((length, width), dtype=np.float32)
    return [q, k, v], q.shape, [np.uint32(length), np.uint32(width), np.float32(decay)]


def reference_impl(q, k, v, length, width, decay):
    distance = np.arange(length)[:, None] - np.arange(length)
    weights = np.where(distance >= 0, np.power(float(decay), np.maximum(distance, 0)), 0)
    return (((q @ k.T) / np.sqrt(float(width)) * weights) @ v).astype(np.float32)


def dispatch_config(case):
    return case["length"] * case["width"], 1, 1, 64, 1, 1, False
