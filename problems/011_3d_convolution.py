"""11. 3D Convolution."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"depth": 2, "rows": 3, "cols": 4, "kd": 1, "kh": 1, "kw": 1},
    {"depth": 3, "rows": 5, "cols": 4, "kd": 3, "kh": 5, "kw": 4},
    {"depth": 9, "rows": 13, "cols": 19, "kd": 2, "kh": 3, "kw": 4},
]


def make_inputs(case, rng):
    depth, rows, cols, kd, kh, kw = (
        case["depth"],
        case["rows"],
        case["cols"],
        case["kd"],
        case["kh"],
        case["kw"],
    )
    volume = rng.standard_normal((depth, rows, cols), dtype=np.float32)
    kernel = rng.standard_normal((kd, kh, kw), dtype=np.float32)
    shape = (depth - kd + 1, rows - kh + 1, cols - kw + 1)
    return [volume, kernel], shape, [np.uint32(v) for v in (*volume.shape, *kernel.shape)]


def reference_impl(volume, kernel):
    windows = np.lib.stride_tricks.sliding_window_view(volume, kernel.shape)
    return np.einsum("ijkabc,abc->ijk", windows, kernel)


def dispatch_config(case):
    return (
        (case["depth"] - case["kd"] + 1)
        * (case["rows"] - case["kh"] + 1)
        * (case["cols"] - case["kw"] + 1),
        1,
        1,
        64,
        1,
        1,
        False,
    )
