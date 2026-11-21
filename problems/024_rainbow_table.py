"""24. Rainbow Table."""

import numpy as np

DTYPE = np.uint32
REFERENCE_USES_SCALARS = True

TEST_CASES = [{"N": n, "rounds": r} for n, r in ((1, 1), (41, 7), (32771, 13))]


def make_inputs(case, rng):
    n, r = case["N"], case["rounds"]
    x = rng.integers(0, 2**31, n, dtype=np.int32)
    return [x], x.shape, [np.uint32(n), np.uint32(r)]


def reference_impl(x, n, rounds):
    words = x.astype(np.uint32)
    for _ in range(rounds):
        hashed = np.full(words.shape, 2166136261, dtype=np.uint32)
        for shift in (0, 8, 16, 24):
            hashed ^= (words >> shift) & 255
            hashed *= np.uint32(16777619)
        words = hashed
    return words


def dispatch_config(case):
    return case["N"], 1, 1, 64, 1, 1, False
