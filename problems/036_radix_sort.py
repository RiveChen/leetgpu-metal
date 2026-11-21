"""36. Radix Sort."""

import numpy as np

DTYPE = np.uint32
RTOL, ATOL = 0, 0
TEST_CASES = [{"n": n} for n in (1, 19, 301, 2048)]


def make_inputs(case, rng):
    n = case["n"]
    keys = rng.integers(0, 2**32, n, dtype=np.uint32)
    if n >= 5:
        keys[:5] = [2**32 - 1, 0, 2**31, 1, 0]
    return [keys], keys.shape, [np.uint32(n)]


def reference_impl(keys):
    return np.sort(keys)


def dispatch_config(case):
    return 1, 1, 1, 1, 1, 1, False


def workspace_bytes(case):
    return 4 * case["n"]
