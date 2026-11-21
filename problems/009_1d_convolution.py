"""9. 1D Convolution."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001

TEST_CASES = [{"N": n, "K": k} for n, k in ((17, 1), (9, 9), (1031, 7), (65539, 13))]


def make_inputs(case, rng):
    n, k = case["N"], case["K"]
    x = rng.standard_normal(n, dtype=np.float32)
    f = rng.standard_normal(k, dtype=np.float32)
    return [x, f], (n - k + 1,), [np.uint32(n), np.uint32(k)]


def reference_impl(x, weights):
    return np.correlate(x, weights, mode="valid")


def dispatch_config(case):
    return case["N"] - case["K"] + 1, 1, 1, 64, 1, 1, False
