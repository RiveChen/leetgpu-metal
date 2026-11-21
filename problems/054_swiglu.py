"""54. Swish-Gated Linear Unit."""

import numpy as np

RTOL, ATOL = 1e-05, 0.0001

TEST_CASES = [{"N": n} for n in (2, 74, 65538)]


def make_inputs(case, rng):
    n = case["N"]
    x = rng.standard_normal(n, dtype=np.float32)
    return [x], (n // 2,), [np.uint32(n)]


def reference_impl(x):
    left, gate = np.split(x, 2)
    return left / (1 + np.exp(-left)) * gate


def dispatch_config(case):
    return case["N"] // 2, 1, 1, 64, 1, 1, False
