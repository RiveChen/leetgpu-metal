"""70. Segmented Exclusive Prefix Sum."""

import numpy as np

RTOL, ATOL = 0.001, 0.001

TEST_CASES = [{"n": 1, "stride": 1}, {"n": 33, "stride": 1}, {"n": 1073, "stride": 13}]


def make_inputs(case, rng):
    n, stride = case["n"], case["stride"]
    x = rng.standard_normal(n, dtype=np.float32)
    flags = (np.arange(n) % stride == 0).astype(np.int32)
    return [x, flags], x.shape, [np.uint32(n)]


def reference_impl(x, flags):
    out = np.zeros_like(x)
    for i in range(1, x.size):
        if not flags[i]:
            out[i] = out[i - 1] + x[i - 1]
    return out


def dispatch_config(case):
    return case["n"], 1, 1, 64, 1, 1, False
