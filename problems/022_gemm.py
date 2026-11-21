"""22. General Matrix Multiplication (GEMM)."""

import numpy as np

DTYPE = np.float16
IN_PLACE = True
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 0.05, 0.05
TEST_CASES = [
    {"m": 16, "n": 16, "k": 16, "alpha": 0.0, "beta": 0.5},
    {"m": 19, "n": 23, "k": 17, "alpha": 1.25, "beta": -0.5},
    {"m": 47, "n": 31, "k": 64, "alpha": -1.0, "beta": 0.0},
]


def make_inputs(case, rng):
    m, n, k, alpha, beta = case["m"], case["n"], case["k"], case["alpha"], case["beta"]
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    c = rng.standard_normal((m, n)).astype(np.float16)
    return (
        [c, a, b],
        c.shape,
        [np.uint32(v) for v in (m, n, k)] + [np.float32(alpha), np.float32(beta)],
    )


def reference_impl(c, a, b, m, n, k, alpha, beta):
    product = a.astype(np.float32) @ b.astype(np.float32)
    return (alpha * product + beta * c.astype(np.float32)).astype(np.float16)


def dispatch_config(case):
    return case["m"] * case["n"], 1, 1, 64, 1, 1, False
