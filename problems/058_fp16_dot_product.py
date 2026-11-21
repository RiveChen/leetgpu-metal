"""58. FP16 Dot Product."""

import numpy as np

DTYPE = np.float16
RTOL, ATOL = 0.05, 0.05
TEST_CASES = [{"n": 1}, {"n": 43}, {"n": 4133}]


def make_inputs(case, rng):
    n = case["n"]
    a = rng.standard_normal(n).astype(np.float16)
    b = rng.standard_normal(n).astype(np.float16)
    return [a, b], (1,), [np.uint32(n)]


def reference_impl(a, b):
    return np.array([np.dot(a.astype(np.float32), b.astype(np.float32))], dtype=np.float16)


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
