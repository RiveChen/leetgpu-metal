"""75. Sparse Matrix-Dense Matrix Multiplication."""

import numpy as np

RTOL, ATOL = 0.001, 0.001

TEST_CASES = [{"m": 1, "n": 1, "k": 1}, {"m": 5, "n": 9, "k": 4}, {"m": 37, "n": 41, "k": 13}]


def make_inputs(case, rng):
    m, n, k = case["m"], case["n"], case["k"]
    a = rng.standard_normal((m, n), dtype=np.float32)
    a[rng.random(a.shape) < 0.7] = 0
    b = rng.standard_normal((n, k), dtype=np.float32)
    return (
        [a, b],
        (m, k),
        [np.uint32(v) for v in (m, n, k, np.count_nonzero(a))],
    )


def reference_impl(a, b):
    return a @ b


def dispatch_config(case):
    return case["m"] * case["k"], 1, 1, 64, 1, 1, False
