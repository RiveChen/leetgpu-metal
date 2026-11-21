"""47. Subarray Sum."""

import numpy as np

DTYPE = np.int32
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"n": 1, "first_i": 0, "last_i": 0},
    {"n": 37, "first_i": 36, "last_i": 36},
    {"n": 2051, "first_i": 1, "last_i": 2049},
]


def make_inputs(case, rng):
    n, first_i, last_i = case["n"], case["first_i"], case["last_i"]
    x = rng.integers(-6, 7, (n,), dtype=np.int32)
    return (
        [x],
        (1,),
        [
            np.uint32(v)
            for v in (
                n,
                first_i,
                last_i,
            )
        ],
    )


def reference_impl(x, n, first_i, last_i):
    return np.array([x[first_i : last_i + 1].sum(dtype=np.int64)], dtype=np.int32)


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
