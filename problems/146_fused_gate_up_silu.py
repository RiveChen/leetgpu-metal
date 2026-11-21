"""146. Fused Gate-Up Projection with SiLU."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"rows": 1, "hidden": 1, "width": 1},
    {"rows": 9, "hidden": 13, "width": 7},
    {"rows": 25, "hidden": 64, "width": 33},
]


def make_inputs(case, rng):
    rows, hidden, width = case["rows"], case["hidden"], case["width"]
    x = rng.standard_normal((rows, hidden), dtype=np.float32)
    weights = rng.standard_normal((hidden, 2 * width), dtype=np.float32) / np.float32(
        np.sqrt(hidden)
    )
    return [x, weights], (rows, width), [np.uint32(v) for v in (rows, hidden, width)]


def reference_impl(x, weights, rows, hidden, width):
    gate, up = np.split(x.astype(np.float64) @ weights, 2, axis=1)
    return (up * gate * np.exp(-np.logaddexp(0, -gate))).astype(np.float32)


def dispatch_config(case):
    return case["rows"] * case["width"], 1, 1, 64, 1, 1, False
