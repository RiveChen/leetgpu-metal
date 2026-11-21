"""125. Scatter Add."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"rows": 1, "width": 1, "groups": 1},
    {"rows": 11, "width": 5, "groups": 19},
    {"rows": 123, "width": 31, "groups": 17},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    rows = case["rows"]
    width = case["width"]
    groups = case["groups"]
    src = rng.standard_normal((rows, width), dtype=np.float32)
    indices = rng.integers(0, groups, rows, dtype=np.int32)
    indices[::3] = 0
    return [src, indices], (groups, width), [np.uint32(v) for v in (rows, width, groups)]


def reference_impl(src, indices, rows, width, groups):
    output = np.zeros((groups, width), dtype=src.dtype)
    np.add.at(output, indices, src)
    return output


def dispatch_config(case):
    return case["groups"] * case["width"], 1, 1, 64, 1, 1, False
