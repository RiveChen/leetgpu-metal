"""43. Count Array Element."""

import numpy as np

DTYPE = np.int32
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [{"n": 1, "target": 99}, {"n": 37, "target": -2}, {"n": 2051, "target": 3}]


def make_inputs(case, rng):
    n, target = case["n"], case["target"]
    x = rng.integers(-6, 7, (n,), dtype=np.int32)
    return [x], (1,), [np.uint32(size) for size in x.shape] + [np.int32(target)]


def reference_impl(x, n, target):
    return np.array([np.count_nonzero(x == target)], dtype=np.int32)


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
