"""144. Layer Norm Backward."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [{"rows": 1, "channels": 1}, {"rows": 9, "channels": 3}, {"rows": 37, "channels": 45}]


def make_inputs(case, rng):
    rows = case["rows"]
    channels = case["channels"]
    x = rng.standard_normal((rows, channels), dtype=np.float32)
    gamma = rng.standard_normal(channels, dtype=np.float32)
    mean = x.mean(axis=1)
    rstd = 1 / np.sqrt(x.var(axis=1) + np.float32(0.0001))
    dy = rng.standard_normal(x.shape, dtype=np.float32)
    return (
        [x, gamma, mean, rstd, dy],
        [(x.shape, np.float32), (gamma.shape, np.float32), (gamma.shape, np.float32)],
        [np.uint32(rows), np.uint32(channels)],
    )


def reference_impl(x, gamma, mean, rstd, dy):
    normalized = (x - mean[:, None]) * rstd[:, None]
    g = dy * gamma
    dx = (
        g
        - g.mean(axis=1, keepdims=True)
        - normalized * (g * normalized).mean(axis=1, keepdims=True)
    ) * rstd[:, None]
    return dx, (dy * normalized).sum(axis=0), dy.sum(axis=0)


def dispatch_config(case):
    return case["channels"], 1, 1, 64, 1, 1, False
