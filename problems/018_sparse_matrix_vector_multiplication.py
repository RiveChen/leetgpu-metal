"""18. Sparse Matrix-Vector Multiplication."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [{"rows": 1, "cols": 1}, {"rows": 19, "cols": 7}, {"rows": 129, "cols": 193}]


def make_inputs(case, rng):
    rows, cols = case["rows"], case["cols"]
    a = rng.standard_normal((rows, cols), dtype=np.float32)
    a[rng.random(a.shape) < 0.65] = 0
    a[0] = 0
    x = rng.standard_normal(cols, dtype=np.float32)
    return [a, x], (rows,), [np.uint32(rows), np.uint32(cols), np.uint32(np.count_nonzero(a))]


def reference_impl(a, x):
    return a @ x


def dispatch_config(case):
    return case["rows"], 1, 1, 64, 1, 1, False
