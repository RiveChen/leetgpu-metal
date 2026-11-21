"""30. Batched Matrix Multiplication."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"batch": 1, "m": 1, "n": 3, "k": 1},
    {"batch": 3, "m": 7, "n": 11, "k": 5},
    {"batch": 5, "m": 33, "n": 19, "k": 31},
]


def make_inputs(case, rng):
    batch, m, n, k = case["batch"], case["m"], case["n"], case["k"]
    a = rng.standard_normal((batch, m, k), dtype=np.float32)
    b = rng.standard_normal((batch, k, n), dtype=np.float32)
    return [a, b], (batch, m, n), [np.uint32(v) for v in (batch, m, n, k)]


def reference_impl(a, b):
    return a @ b


def dispatch_config(case):
    return case["batch"] * case["m"] * case["n"], 1, 1, 64, 1, 1, False
