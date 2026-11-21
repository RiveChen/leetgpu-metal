"""29. Top K Selection."""

import numpy as np

REFERENCE_USES_SCALARS = True
RTOL, ATOL = 0.0, 1e-05
TEST_CASES = [{"n": 1, "k": 1}, {"n": 19, "k": 19}, {"n": 259, "k": 17}]


def make_inputs(case, rng):
    n, k = case["n"], case["k"]
    values = rng.integers(-12, 13, n).astype(np.float32)
    values[: min(3, n)] = 15
    return [values], (k,), [np.uint32(n), np.uint32(k)]


def reference_impl(values, n, k):
    return np.sort(values)[::-1][:k].copy()


def dispatch_config(case):
    return case["n"], 1, 1, 64, 1, 1, False
