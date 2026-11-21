"""62. Value Clipping."""

import numpy as np

REFERENCE_USES_SCALARS = True

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [
    {"N": n, "lo": lo, "hi": hi}
    for n, lo, hi in ((13, 0.25, 0.25), (73, -2, -0.5), (65539, -0.3, 1.4))
]


def make_inputs(case, rng):
    n = case["N"]
    x = rng.standard_normal(n, dtype=np.float32)
    lo, hi = np.float32(case["lo"]), np.float32(case["hi"])
    return [x], x.shape, [lo, hi, np.uint32(n)]


def reference_impl(x, lo, hi, n):
    return np.clip(x, lo, hi)


def dispatch_config(case):
    return case["N"], 1, 1, 64, 1, 1, False
