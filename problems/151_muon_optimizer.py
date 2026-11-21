"""151. Muon Optimizer."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"rows": 1, "cols": 1, "momentum": 0.0},
    {"rows": 3, "cols": 7, "momentum": 0.4},
    {"rows": 11, "cols": 5, "momentum": 0.8},
    {"rows": 8, "cols": 8, "momentum": 0.95},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    rows = case["rows"]
    cols = case["cols"]
    momentum = case["momentum"]
    g = rng.standard_normal((rows, cols), dtype=np.float32)
    buf = rng.standard_normal(g.shape, dtype=np.float32)
    return [g, buf], g.shape, [np.uint32(rows), np.uint32(cols), np.float32(momentum)]


def reference_impl(g, buf, rows, cols, momentum):
    x = g + momentum * buf
    x = x / (np.linalg.norm(x) + np.float32(1e-7))
    tall = rows > cols
    if tall:
        x = x.T
    for _ in range(5):
        gram = x @ x.T
        x = 3.4445 * x + (-4.775 * gram + 2.0315 * (gram @ gram)) @ x
    return (x.T if tall else x) * np.float32(np.sqrt(max(rows, cols)))


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    rows, cols = case["rows"], case["cols"]
    return 4 * (rows * cols + 2 * min(rows, cols) ** 2)
