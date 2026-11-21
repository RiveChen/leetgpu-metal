"""122. Hamming Top-K Search."""

import numpy as np

DTYPE = np.int32
RTOL, ATOL = 0, 0
TEST_CASES = [{"rows": 1, "top": 1}, {"rows": 13, "top": 13}, {"rows": 129, "top": 11}]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    rows = case["rows"]
    top = case["top"]
    codes = rng.integers(-(2**31), 2**31, (rows, 8), dtype=np.int32)
    query = rng.integers(-(2**31), 2**31, 8, dtype=np.int32)
    codes[::4] = query
    return [codes, query], (top,), [np.uint32(rows), np.uint32(top)]


def reference_impl(codes, query, rows, top):
    bits = np.bitwise_xor(codes, query).view(np.uint8)
    distances = np.unpackbits(bits, axis=1).sum(axis=1)
    return np.argsort(distances, kind="stable")[:top].astype(np.int32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1


def workspace_bytes(case):
    return 2 * case["rows"]
