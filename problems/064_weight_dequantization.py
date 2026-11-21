"""64. Weight Dequantization."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"m": 1, "n": 1, "tile": 1},
    {"m": 13, "n": 21, "tile": 8},
    {"m": 64, "n": 33, "tile": 4},
]


def make_inputs(case, rng):
    m, n, tile = case["m"], case["n"], case["tile"]
    x = rng.integers(-127, 128, (m, n)).astype(np.float32)
    scale = rng.uniform(0.01, 2, ((m + tile - 1) // tile, (n + tile - 1) // tile)).astype(
        np.float32
    )
    return (
        [x, scale],
        x.shape,
        [np.uint32(v) for v in (m, n, tile)],
    )


def reference_impl(x, scale, m, n, tile):
    expanded = np.repeat(np.repeat(scale, tile, axis=0), tile, axis=1)
    return x * expanded[:m, :n]


def dispatch_config(case):
    return case["m"] * case["n"], 1, 1, 64, 1, 1, False
