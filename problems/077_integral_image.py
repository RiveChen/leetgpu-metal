"""77. Integral Image."""

import numpy as np

RTOL, ATOL = 0.01, 0.01

TEST_CASES = [{"h": 1, "w": 13}, {"h": 9, "w": 1}, {"h": 57, "w": 37}]


def make_inputs(case, rng):
    h, w = case["h"], case["w"]
    x = rng.standard_normal((h, w), dtype=np.float32)
    return (
        [x],
        x.shape,
        [np.uint32(v) for v in (h, w)],
    )


def reference_impl(x):
    return x.cumsum(axis=0, dtype=np.float64).cumsum(axis=1).astype(np.float32)


def dispatch_config(case):
    return case["h"], 1, 1, 64, 1, 1, False
