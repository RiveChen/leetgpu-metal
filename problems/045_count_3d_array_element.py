"""45. Count 3D Array Element."""

import numpy as np

DTYPE = np.int32
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"depth": 1, "rows": 3, "cols": 5, "target": 99},
    {"depth": 5, "rows": 9, "cols": 3, "target": -2},
    {"depth": 11, "rows": 17, "cols": 23, "target": 3},
]


def make_inputs(case, rng):
    depth, rows, cols, target = case["depth"], case["rows"], case["cols"], case["target"]
    x = rng.integers(
        -6,
        7,
        (
            depth,
            rows,
            cols,
        ),
        dtype=np.int32,
    )
    return [x], (1,), [np.uint32(size) for size in x.shape] + [np.int32(target)]


def reference_impl(x, depth, rows, cols, target):
    return np.array([np.count_nonzero(x == target)], dtype=np.int32)


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
