"""105. Group Normalization."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "channels": 2, "height": 1, "width": 1, "groups": 2, "eps": 1e-05},
    {"batch": 3, "channels": 6, "height": 2, "width": 5, "groups": 3, "eps": 1e-05},
    {"batch": 3, "channels": 12, "height": 7, "width": 4, "groups": 4, "eps": 1e-05},
]


def make_inputs(case, rng):
    batch = case["batch"]
    channels = case["channels"]
    height = case["height"]
    width = case["width"]
    groups = case["groups"]
    eps = case["eps"]
    x = rng.standard_normal((batch, channels, height, width), dtype=np.float32)
    gamma = rng.uniform(0.1, 10, channels).astype(np.float32)
    beta = rng.standard_normal(channels, dtype=np.float32)
    return (
        [x, gamma, beta],
        x.shape,
        [
            np.uint32(batch),
            np.uint32(channels),
            np.uint32(height),
            np.uint32(width),
            np.uint32(groups),
            np.float32(eps),
        ],
    )


def reference_impl(x, gamma, beta, batch, channels, height, width, groups, eps):
    grouped = x.reshape(batch, groups, -1)
    centered = grouped - grouped.mean(axis=-1, keepdims=True)
    normalized = centered / np.sqrt(grouped.var(axis=-1, keepdims=True) + eps)
    return normalized.reshape(x.shape) * gamma[None, :, None, None] + beta[None, :, None, None]


def dispatch_config(case):
    return case["batch"] * case["groups"], 1, 1, 64, 1, 1, False
