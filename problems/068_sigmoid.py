"""68. Sigmoid Activation."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [{"N": n} for n in (1, 37, 65539)]


def make_inputs(case, rng):
    n = case["N"]
    x = rng.standard_normal(n, dtype=np.float32)
    return [x], x.shape, [np.uint32(n)]


def reference_impl(x):
    return 1 / (1 + np.exp(-x))


def dispatch_config(case):
    return case["N"], 1, 1, 64, 1, 1, False
