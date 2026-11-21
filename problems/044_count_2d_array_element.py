"""44. Count 2D Array Element."""

import numpy as np

DTYPE = np.int32
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"rows": 1, "cols": 7, "target": 99},
    {"rows": 7, "cols": 11, "target": -2},
    {"rows": 47, "cols": 63, "target": 3},
]


def make_inputs(case, rng):
    rows, cols, target = case["rows"], case["cols"], case["target"]
    x = rng.integers(
        -6,
        7,
        (
            rows,
            cols,
        ),
        dtype=np.int32,
    )
    return [x], (1,), [np.uint32(size) for size in x.shape] + [np.int32(target)]


def reference_impl(x, rows, cols, target):
    return np.array([np.count_nonzero(x == target)], dtype=np.int32)


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
