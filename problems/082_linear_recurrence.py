"""82. Linear Recurrence."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [{"batch": 1, "length": 1}, {"batch": 3, "length": 47}, {"batch": 13, "length": 137}]


def make_inputs(case, rng):
    batch, length = case["batch"], case["length"]
    a = rng.uniform(-0.8, 0.8, (batch, length)).astype(np.float32)
    x = rng.standard_normal(a.shape, dtype=np.float32)
    return (
        [a, x],
        x.shape,
        [np.uint32(v) for v in (batch, length)],
    )


def reference_impl(a, x):
    h = x.copy()
    for t in range(1, x.shape[1]):
        h[:, t] += a[:, t] * h[:, t - 1]
    return h


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, False
