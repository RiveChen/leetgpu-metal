"""37. Matrix Power."""

import numpy as np

REFERENCE_USES_SCALARS = True
RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"n": 3, "power": 1},
    {"n": 9, "power": 2},
    {"n": 13, "power": 5},
    {"n": 5, "power": 20},
]


def make_inputs(case, rng):
    n, power = case["n"], case["power"]
    a = rng.uniform(-1, 1, (n, n)).astype(np.float32) / np.float32(np.sqrt(n))
    return [a], a.shape, [np.uint32(n), np.uint32(power)]


def reference_impl(a, n, power):
    return np.linalg.matrix_power(a.astype(np.float64), int(power)).astype(np.float32)


def dispatch_config(case):
    return case["n"], 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 4 * case["n"] ** 2
