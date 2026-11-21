"""19. Reverse Array."""

import numpy as np

IN_PLACE = True

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [{"N": n} for n in (31, 64, 65539, 1)]


def make_inputs(case, rng):
    n = case["N"]
    x = rng.standard_normal(n, dtype=np.float32)
    return [x], x.shape, [np.uint32(n)]


def reference_impl(x):
    return x[::-1].copy()


def dispatch_config(case):
    return case["N"], 1, 1, 64, 1, 1, False
