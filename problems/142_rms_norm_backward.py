"""142. RMS Norm Backward."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [
    {"rows": 1, "channels": 1, "eps": 0.01},
    {"rows": 3, "channels": 7, "eps": 1e-05},
    {"rows": 31, "channels": 43, "eps": 0.0001},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    rows = case["rows"]
    channels = case["channels"]
    eps = case["eps"]
    x = rng.standard_normal((rows, channels), dtype=np.float32)
    weight = rng.standard_normal(channels, dtype=np.float32)
    dy = rng.standard_normal(x.shape, dtype=np.float32)
    return (
        [x, weight, dy],
        [(x.shape, np.float32), (weight.shape, np.float32)],
        [np.uint32(rows), np.uint32(channels), np.float32(eps)],
    )


def reference_impl(x, weight, dy, rows, channels, eps):
    inv = 1 / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + eps)
    normalized = x * inv
    g = dy * weight
    return inv * (g - normalized * np.mean(normalized * g, axis=1, keepdims=True)), np.sum(
        dy * normalized, axis=0
    )


def dispatch_config(case):
    return case["channels"], 1, 1, 64, 1, 1, False
