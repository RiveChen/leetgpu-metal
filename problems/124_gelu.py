"""124. GELU Activation."""

import math

import numpy as np

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [{"N": n} for n in (1, 37, 65539)]


def make_inputs(case, rng):
    n = case["N"]
    x = rng.standard_normal(n, dtype=np.float32)
    return [x], x.shape, [np.uint32(n)]


def reference_impl(gate):
    return (
        0.5
        * gate
        * np.array([math.erfc(-float(value) / math.sqrt(2)) for value in gate], dtype=np.float32)
    )


def dispatch_config(case):
    return case["N"], 1, 1, 64, 1, 1, False
