"""65. Gaussian Error Gated Linear Unit."""

import math

import numpy as np

RTOL, ATOL = 0.0001, 0.0001

TEST_CASES = [{"N": n} for n in (2, 74, 65538)]


def make_inputs(case, rng):
    n = case["N"]
    x = rng.standard_normal(n, dtype=np.float32)
    return [x], (n // 2,), [np.uint32(n)]


def reference_impl(x):
    left, gate = np.split(x, 2)
    return left * (
        0.5
        * gate
        * np.array([math.erfc(-float(value) / math.sqrt(2)) for value in gate], dtype=np.float32)
    )


def dispatch_config(case):
    return case["N"] // 2, 1, 1, 64, 1, 1, False
