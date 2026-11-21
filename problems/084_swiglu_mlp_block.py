"""84. SwiGLU MLP Block."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001

TEST_CASES = [
    {"rows": 1, "width": 1, "hidden": 3},
    {"rows": 5, "width": 13, "hidden": 19},
    {"rows": 31, "width": 21, "hidden": 37},
]


def make_inputs(case, rng):
    rows, width, hidden = case["rows"], case["width"], case["hidden"]
    x = rng.standard_normal((rows, width), dtype=np.float32)
    gate = rng.standard_normal((width, hidden), dtype=np.float32) / np.float32(np.sqrt(width))
    up = rng.standard_normal(gate.shape, dtype=np.float32)
    down = rng.standard_normal((hidden, width), dtype=np.float32)
    return (
        [x, gate, up, down],
        x.shape,
        [np.uint32(v) for v in (rows, width, hidden)],
    )


def reference_impl(x, gate, up, down):
    g = x @ gate
    return ((g / (1 + np.exp(-g))) * (x @ up)) @ down


def dispatch_config(case):
    return case["rows"] * case["width"], 1, 1, 64, 1, 1, False
