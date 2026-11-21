"""38. Nearest Neighbor."""

import numpy as np

DTYPE = np.int32
RTOL, ATOL = 0, 0
TEST_CASES = [{"n": 2}, {"n": 19}, {"n": 257}]


def make_inputs(case, rng):
    n = case["n"]
    points = rng.standard_normal((n, 3), dtype=np.float32)
    if n >= 3:
        points[:3] = points[0]
    return [points], (n,), [np.uint32(n)]


def reference_impl(points):
    differences = points[:, None, :] - points[None, :, :]
    distances = (differences**2).sum(axis=-1)
    np.fill_diagonal(distances, np.inf)
    return np.argmin(distances, axis=1).astype(np.int32)


def dispatch_config(case):
    return case["n"], 1, 1, 64, 1, 1, False
