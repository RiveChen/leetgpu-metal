"""99. Grouped GEMM."""

import numpy as np


RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"groups": 1, "rows": 1, "inner": 3, "cols": 2},
    {"groups": 5, "rows": 7, "inner": 3, "cols": 9},
    {"groups": 7, "rows": 71, "inner": 17, "cols": 13},
]


def make_inputs(case, rng):
    groups = case["groups"]
    rows = case["rows"]
    inner = case["inner"]
    cols = case["cols"]
    a = rng.uniform(-1, 1, (rows, inner)).astype(np.float32)
    b = rng.uniform(-1, 1, (groups, inner, cols)).astype(np.float32)
    offsets = np.sort(np.r_[0, rng.integers(0, rows + 1, groups - 1), rows]).astype(np.int32)
    return (
        [a, b, offsets],
        (rows, cols),
        [np.uint32(groups), np.uint32(rows), np.uint32(inner), np.uint32(cols)],
    )


def reference_impl(a, b, offsets, groups, rows, inner, cols):
    result = np.empty((rows, cols), np.float32)
    for group, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        result[start:end] = a[start:end] @ b[group]
    return result


def dispatch_config(case):
    return case["rows"] * case["cols"], 1, 1, 64, 1, 1, False
