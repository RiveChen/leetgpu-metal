"""40. Batch Normalization."""

import numpy as np

REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"samples": 1, "channels": 3},
    {"samples": 17, "channels": 11},
    {"samples": 131, "channels": 19},
]


def make_inputs(case, rng):
    samples, channels = case["samples"], case["channels"]
    x = rng.standard_normal((samples, channels), dtype=np.float32)
    x[:, 0] = 2
    gamma = rng.standard_normal(channels, dtype=np.float32)
    beta = rng.standard_normal(channels, dtype=np.float32)
    return [x, gamma, beta], x.shape, [np.uint32(samples), np.uint32(channels), np.float32(1e-5)]


def reference_impl(x, gamma, beta, samples, channels, eps):
    centered = x.astype(np.float64) - x.mean(axis=0, dtype=np.float64)
    variance = np.mean(centered**2, axis=0)
    return (gamma * centered / np.sqrt(variance + eps) + beta).astype(np.float32)


def dispatch_config(case):
    return case["channels"], 1, 1, 64, 1, 1, False
