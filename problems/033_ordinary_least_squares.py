"""33. Ordinary Least Squares."""

import numpy as np

RTOL, ATOL = 0.01, 0.01
TEST_CASES = [
    {"samples": 23, "features": 1},
    {"samples": 61, "features": 5},
    {"samples": 167, "features": 11},
]


def make_inputs(case, rng):
    samples, features = case["samples"], case["features"]
    x = rng.standard_normal((samples, features), dtype=np.float32)
    signal = x @ rng.uniform(-0.4, 0.4, features).astype(np.float32)
    y = signal + 0.03 * rng.standard_normal(samples, dtype=np.float32)
    return [x, y], (features,), [np.uint32(samples), np.uint32(features)]


def reference_impl(x, y):
    return np.linalg.lstsq(x.astype(np.float64), y, rcond=None)[0].astype(np.float32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 4 * case["features"] ** 2
