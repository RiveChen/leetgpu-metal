"""32. INT8 Quantized MatMul."""

import numpy as np

DTYPE = np.int8
REFERENCE_USES_SCALARS = True
RTOL, ATOL = 0, 0
TEST_CASES = [{"m": 1, "n": 5, "k": 7}, {"m": 17, "n": 1, "k": 65}, {"m": 37, "n": 29, "k": 19}]


def make_inputs(case, rng):
    m, n, k = case["m"], case["n"], case["k"]
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (k, n), dtype=np.int8)
    scales = [np.float32(s) for s in (0.03125, 0.125, 0.5)]
    offsets = [np.int32(v) for v in (-3, 5, 9)]
    return [a, b], (m, n), [np.uint32(v) for v in (m, n, k)] + scales + offsets


def reference_impl(a, b, m, n, k, scale_a, scale_b, scale_c, zero_a, zero_b, zero_c):
    dot = (a.astype(np.int32) - zero_a) @ (b.astype(np.int32) - zero_b)
    scaled = dot.astype(np.float32) * (scale_a * scale_b / scale_c)
    return np.clip(np.rint(scaled) + zero_c, -128, 127).astype(np.int8)


def dispatch_config(case):
    return case["m"] * case["n"], 1, 1, 64, 1, 1, False
