"""69. 2D Jacobi Stencil."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [{"h": 2, "w": 9}, {"h": 5, "w": 7}, {"h": 31, "w": 37}]


def make_inputs(case, rng):
    h, w = case["h"], case["w"]
    x = rng.standard_normal((h, w), dtype=np.float32)
    return (
        [x],
        x.shape,
        [np.uint32(v) for v in (h, w)],
    )


def reference_impl(x):
    out = x.copy()
    out[1:-1, 1:-1] = (x[:-2, 1:-1] + x[2:, 1:-1] + x[1:-1, :-2] + x[1:-1, 2:]) * 0.25
    return out


def dispatch_config(case):
    return case["h"] * case["w"], 1, 1, 64, 1, 1, False
