"""35. Monte Carlo Integration."""

import numpy as np

REFERENCE_USES_SCALARS = True
RTOL, ATOL = 0.01, 0.01
TEST_CASES = [
    {"n": 1, "a": -1, "b": 2},
    {"n": 73, "a": -3, "b": 0},
    {"n": 1027, "a": 0.2, "b": 1.8},
]


def make_inputs(case, rng):
    n, a, b = case["n"], case["a"], case["b"]
    x = rng.uniform(a, b, n)
    y = (x * x + np.sin(3 * x)).astype(np.float32)
    return [y], (1,), [np.float32(a), np.float32(b), np.uint32(n)]


def reference_impl(y, a, b, n):
    return np.array([(b - a) * y.mean(dtype=np.float64)], dtype=np.float32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
