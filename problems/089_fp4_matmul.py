"""89. NVFP4 Matrix Multiplication."""

import numpy as np

RTOL, ATOL = 0.05, 0.1
REFERENCE_USES_SCALARS = True
DTYPE = np.float16

TEST_CASES = [
    {"alpha": 1.0, "rows": 1, "cols": 1, "inner": 16},
    {"alpha": 0.3, "rows": 3, "cols": 7, "inner": 32},
    {"alpha": 1.5, "rows": 9, "cols": 11, "inner": 80},
]


def make_inputs(case, rng):
    alpha = case["alpha"]
    rows = case["rows"]
    cols = case["cols"]
    inner = case["inner"]
    shape = (rows, cols)
    x = rng.integers(0, 256, (rows, inner // 2), dtype=np.uint8)
    w = rng.integers(0, 256, (cols, inner // 2), dtype=np.uint8)
    xs = rng.integers(1, 65, (rows, inner // 16), dtype=np.uint8)
    ws = rng.integers(1, 65, (cols, inner // 16), dtype=np.uint8)
    return (
        [x, xs, w, ws],
        shape,
        [np.float32(alpha), np.uint32(rows), np.uint32(cols), np.uint32(inner)],
    )


def reference_impl(x, xs, w, ws, alpha, rows, cols, inner):
    def decode(packed, scales):
        codes = np.stack((packed >> 4, packed & 15), axis=-1).reshape(len(packed), -1)
        magnitude = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)[codes & 7]
        values = np.where(codes & 8, -magnitude, magnitude)
        exponent = (scales.astype(np.int32) >> 3) & 15
        mantissa = (scales & 7).astype(np.float32) / 8
        scale = np.ldexp((exponent != 0) + mantissa, np.maximum(exponent, 1) - 7)
        scale = np.where(scales & 128, -scale, scale)
        return values * np.repeat(scale, 16, axis=1)

    return (alpha * (decode(x, xs) @ decode(w, ws).T)).astype(np.float16)


def dispatch_config(case):
    return case["rows"] * case["cols"], 1, 1, 64, 1, 1, False
