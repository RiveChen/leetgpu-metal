"""83. Fused Residual Add and RMS Norm."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"rows": 1, "width": 1, "eps": 0.001},
    {"rows": 7, "width": 31, "eps": 0.01},
    {"rows": 41, "width": 65, "eps": 1e-05},
]


def make_inputs(case, rng):
    rows, width, eps = case["rows"], case["width"], case["eps"]
    x = rng.standard_normal((rows, width), dtype=np.float32)
    res = rng.standard_normal(x.shape, dtype=np.float32)
    res[0] = -x[0]
    weight = rng.standard_normal(width, dtype=np.float32)
    return [x, res, weight], x.shape, [np.uint32(rows), np.uint32(width), np.float32(eps)]


def reference_impl(x, res, weight, rows, width, eps):
    z = x + res
    return weight * z / np.sqrt(np.mean(z * z, axis=-1, keepdims=True) + eps)


def dispatch_config(case):
    return case["rows"], 1, 1, 64, 1, 1, False
