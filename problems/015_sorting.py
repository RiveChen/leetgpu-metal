"""15. Sorting."""

import numpy as np

IN_PLACE = True
RTOL, ATOL = 0, 0
TEST_CASES = [{"n": n} for n in (47, 1, 256, 1031)]


def make_inputs(case, rng):
    n = case["n"]
    values = rng.standard_normal(n, dtype=np.float32)
    values[::7] = 0
    return [values], values.shape, [np.uint32(n)]


def reference_impl(values):
    return np.sort(values)


def dispatch_config(case):
    return 1, 1, 1, 1, 1, 1, False
