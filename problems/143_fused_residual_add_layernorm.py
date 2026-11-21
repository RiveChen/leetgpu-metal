"""143. Fused Residual Add and LayerNorm with Saved Statistics."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"rows": 1, "width": 1, "offset": 0, "eps": 0.1},
    {"rows": 7, "width": 13, "offset": 0, "eps": 1e-05},
    {"rows": 19, "width": 65, "offset": 512, "eps": 0.0001},
]


def make_inputs(case, rng):
    rows, width, offset, eps = case["rows"], case["width"], case["offset"], case["eps"]
    x = rng.standard_normal((rows, width), dtype=np.float32) + np.float32(offset)
    residual = rng.standard_normal(x.shape, dtype=np.float32)
    gamma, beta = rng.standard_normal((2, width), dtype=np.float32)
    outputs = [(s, np.float32) for s in (x.shape, x.shape, (rows,), (rows,))]
    return [x, residual, gamma, beta], outputs, [np.uint32(rows), np.uint32(width), np.float32(eps)]


def reference_impl(x, residual, gamma, beta, rows, width, eps):
    h = x + residual
    data = h.astype(np.float64)
    means = data.mean(axis=1)
    centered = data - means[:, None]
    inv = 1 / np.sqrt(np.mean(centered**2, axis=1) + eps)
    return (
        (centered * inv[:, None] * gamma + beta).astype(np.float32),
        h,
        means.astype(np.float32),
        inv.astype(np.float32),
    )


def dispatch_config(case):
    return case["rows"], 1, 1, 64, 1, 1, False
