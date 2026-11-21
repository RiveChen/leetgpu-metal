"""81. INT4 Weight-Only Quantized MatMul."""

import numpy as np

RTOL, ATOL = 0.01, 0.01
DTYPE = np.float16
TEST_CASES = [
    {"m": 1, "n": 1, "k": 8, "group": 4},
    {"m": 7, "n": 5, "k": 16, "group": 16},
    {"m": 13, "n": 17, "k": 48, "group": 8},
]


def make_inputs(case, rng):
    m, n, k, group = case["m"], case["n"], case["k"], case["group"]
    x = rng.standard_normal((m, k)).astype(np.float16)
    packed = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scale = rng.uniform(0.05, 0.5, (n, k // group)).astype(np.float16)
    return (
        [x, packed, scale],
        (m, n),
        [np.uint32(v) for v in (m, n, k, group)],
    )


def reference_impl(x, packed, scale):
    values = (
        np.stack((packed >> 4, packed & 15), axis=-1).reshape(len(packed), -1).astype(np.float32)
        - 8
    )
    weights = values * np.repeat(scale, values.shape[1] // scale.shape[1], axis=1)
    return (x.astype(np.float32) @ weights.T).astype(np.float16)


def dispatch_config(case):
    return case["m"] * case["n"], 1, 1, 64, 1, 1, False
