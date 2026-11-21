"""48. 2D Subarray Sum."""

import numpy as np

DTYPE = np.int32
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"rows": 1, "cols": 7, "first_row": 0, "last_row": 0, "first_col": 0, "last_col": 6},
    {"rows": 7, "cols": 11, "first_row": 6, "last_row": 6, "first_col": 10, "last_col": 10},
    {"rows": 47, "cols": 63, "first_row": 1, "last_row": 45, "first_col": 1, "last_col": 61},
]


def make_inputs(case, rng):
    rows, cols, first_row, last_row, first_col, last_col = (
        case["rows"],
        case["cols"],
        case["first_row"],
        case["last_row"],
        case["first_col"],
        case["last_col"],
    )
    x = rng.integers(
        -6,
        7,
        (
            rows,
            cols,
        ),
        dtype=np.int32,
    )
    return (
        [x],
        (1,),
        [
            np.uint32(v)
            for v in (
                rows,
                cols,
                first_row,
                last_row,
                first_col,
                last_col,
            )
        ],
    )


def reference_impl(x, rows, cols, first_row, last_row, first_col, last_col):
    return np.array(
        [x[first_row : last_row + 1, first_col : last_col + 1].sum(dtype=np.int64)], dtype=np.int32
    )


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
