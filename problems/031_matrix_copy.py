"""31. Matrix Copy."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [{"N": n} for n in (1, 19, 513)]


def make_inputs(case, rng):
    n = case["N"]
    x = rng.standard_normal((n, n), dtype=np.float32)
    return [x], x.shape, [np.uint32(n)]


def reference_impl(x):
    return x.copy()


def dispatch_config(case):
    return case["N"] ** 2, 1, 1, 64, 1, 1, False
