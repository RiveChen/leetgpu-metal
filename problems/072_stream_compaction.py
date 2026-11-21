"""72. Stream Compaction."""

import numpy as np

RTOL, ATOL = 0.0, 0.0

TEST_CASES = [{"n": 9, "shift": -12}, {"n": 37, "shift": 12}, {"n": 1073, "shift": 0}]


def make_inputs(case, rng):
    n, shift = case["n"], case["shift"]
    x = (rng.integers(-8, 9, n) + shift).astype(np.float32)
    return [x], x.shape, [np.uint32(n)]


def reference_impl(x):
    positive = x[x > 0]
    return np.pad(positive, (0, len(x) - len(positive)))


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
