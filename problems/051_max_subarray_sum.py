"""51. Max Subarray Sum."""

import numpy as np

DTYPE = np.int32
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [{"n": 17, "window": 1}, {"n": 23, "window": 23}, {"n": 139, "window": 9}]


def make_inputs(case, rng):
    n, window = case["n"], case["window"]
    x = rng.integers(-10, 11, n, dtype=np.int32)
    if window == 1:
        x = -np.maximum(1, np.abs(x))
    return [x], (1,), [np.uint32(n), np.uint32(window)]


def reference_impl(x, n, window):
    windows = np.lib.stride_tricks.sliding_window_view(x, int(window))
    return np.array([windows.sum(axis=-1).max()], dtype=np.int32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
