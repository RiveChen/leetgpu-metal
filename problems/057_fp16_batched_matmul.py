"""57. FP16 Batched Matrix Multiplication."""

import numpy as np

DTYPE = np.float16
RTOL, ATOL = 0.05, 0.05
TEST_CASES = [
    {"batch": 1, "m": 2, "n": 1, "k": 3},
    {"batch": 3, "m": 17, "n": 11, "k": 13},
    {"batch": 5, "m": 31, "n": 19, "k": 23},
]


def make_inputs(case, rng):
    batch, m, n, k = case["batch"], case["m"], case["n"], case["k"]
    a = rng.standard_normal((batch, m, k)).astype(np.float16)
    b = rng.standard_normal((batch, k, n)).astype(np.float16)
    return [a, b], (batch, m, n), [np.uint32(v) for v in (batch, m, n, k)]


def reference_impl(a, b):
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(np.float16)


def dispatch_config(case):
    return case["batch"] * case["m"] * case["n"], 1, 1, 64, 1, 1, False
