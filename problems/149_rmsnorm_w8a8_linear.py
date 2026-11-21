"""149. Fused RMSNorm and W8A8 Linear."""

import numpy as np

RTOL, ATOL = 0.01, 0.01
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"rows": 1, "cols": 1, "width": 1},
    {"rows": 7, "cols": 9, "width": 13},
    {"rows": 17, "cols": 33, "width": 129},
]


def make_inputs(case, rng):
    rows, cols, width = case["rows"], case["cols"], case["width"]
    x = rng.standard_normal((rows, width), dtype=np.float32)
    if rows > 1:
        x[0] = 0
        x[-1, 0] *= 30
    gamma = rng.uniform(0.3, 1.8, width).astype(np.float32)
    weights = rng.integers(-127, 128, (cols, width), dtype=np.int8)
    scales = rng.uniform(0.5, 1.5, cols).astype(np.float32) / (127 * np.float32(np.sqrt(width)))
    return [x, gamma, weights, scales], (rows, cols), [np.uint32(v) for v in (rows, cols, width)]


def reference_impl(x, gamma, weights, scales, rows, cols, width):
    data = x.astype(np.float64)
    norm = data * gamma / np.sqrt(np.mean(data**2, axis=1, keepdims=True) + 1e-6)
    step = np.max(np.abs(norm), axis=1, keepdims=True) / 127
    quant = np.rint(norm / np.where(step > 0, step, 1)).astype(np.int32)
    return ((quant @ weights.astype(np.int32).T) * step * scales).astype(np.float32)


def dispatch_config(case):
    return case["rows"] * case["cols"], 1, 1, 64, 1, 1, False
