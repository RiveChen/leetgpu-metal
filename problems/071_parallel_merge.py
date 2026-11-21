"""71. Parallel Merge."""

import numpy as np

RTOL, ATOL = 0.0, 0.0

TEST_CASES = [{"m": 1, "n": 1}, {"m": 37, "n": 5}, {"m": 93, "n": 137}]


def make_inputs(case, rng):
    m, n = case["m"], case["n"]
    a = np.sort(rng.integers(-12, 13, m).astype(np.float32))
    b = np.sort(rng.integers(-12, 13, n).astype(np.float32))
    return (
        [a, b],
        (m + n,),
        [np.uint32(v) for v in (m, n)],
    )


def reference_impl(a, b):
    return np.sort(np.concatenate([a, b]))


def dispatch_config(case):
    return case["m"] + case["n"], 1, 1, 64, 1, 1, False
