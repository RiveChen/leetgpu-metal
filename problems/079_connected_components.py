"""79. Connected Components."""

import numpy as np

RTOL, ATOL = 0, 0
DTYPE = np.int32
REFERENCE_USES_SCALARS = True
TEST_CASES = [{"n": 5, "edges": 0}, {"n": 13, "edges": 9}, {"n": 129, "edges": 223}]


def make_inputs(case, rng):
    n, edges = case["n"], case["edges"]
    src = rng.integers(0, n, edges, dtype=np.int32)
    dst = rng.integers(0, n, edges, dtype=np.int32)
    return (
        [src, dst],
        (n,),
        [np.uint32(v) for v in (n, edges)],
    )


def reference_impl(src, dst, n, edges):
    groups = np.arange(n, dtype=np.int32)
    for left, right in zip(src, dst, strict=True):
        a, b = groups[left], groups[right]
        groups[groups == max(a, b)] = min(a, b)
    return groups


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
