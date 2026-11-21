"""49. 3D Subarray Sum."""

import numpy as np

DTYPE = np.int32
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {
        "depth": 1,
        "rows": 3,
        "cols": 5,
        "first_z": 0,
        "last_z": 0,
        "first_row": 0,
        "last_row": 2,
        "first_col": 0,
        "last_col": 4,
    },
    {
        "depth": 5,
        "rows": 9,
        "cols": 3,
        "first_z": 4,
        "last_z": 4,
        "first_row": 8,
        "last_row": 8,
        "first_col": 2,
        "last_col": 2,
    },
    {
        "depth": 11,
        "rows": 17,
        "cols": 23,
        "first_z": 1,
        "last_z": 9,
        "first_row": 1,
        "last_row": 15,
        "first_col": 1,
        "last_col": 21,
    },
]


def make_inputs(case, rng):
    depth, rows, cols, first_z, last_z, first_row, last_row, first_col, last_col = (
        case["depth"],
        case["rows"],
        case["cols"],
        case["first_z"],
        case["last_z"],
        case["first_row"],
        case["last_row"],
        case["first_col"],
        case["last_col"],
    )
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
    return (
        [x],
        (1,),
        [
            np.uint32(v)
            for v in (
                depth,
                rows,
                cols,
                first_z,
                last_z,
                first_row,
                last_row,
                first_col,
                last_col,
            )
        ],
    )


def reference_impl(x, depth, rows, cols, first_z, last_z, first_row, last_row, first_col, last_col):
    return np.array(
        [
            x[first_z : last_z + 1, first_row : last_row + 1, first_col : last_col + 1].sum(
                dtype=np.int64
            )
        ],
        dtype=np.int32,
    )


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
