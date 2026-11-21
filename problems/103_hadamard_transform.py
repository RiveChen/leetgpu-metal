"""103. Hadamard Transform."""

import numpy as np


RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [{"batch": 2, "width": 1}, {"batch": 3, "width": 8}, {"batch": 5, "width": 128}]


def make_inputs(case, rng):
    batch = case["batch"]
    width = case["width"]
    x = rng.standard_normal((batch, width), dtype=np.float32)
    return [x], x.shape, [np.uint32(batch), np.uint32(width)]


def reference_impl(x, batch, width):
    result = x.copy()
    span = 1
    while span < width:
        blocks = result.reshape(batch, -1, 2, span)
        low, high = blocks[:, :, 0].copy(), blocks[:, :, 1].copy()
        blocks[:, :, 0], blocks[:, :, 1] = low + high, low - high
        span *= 2
    return result / np.float32(np.sqrt(width))


def dispatch_config(case):
    return case["batch"] * case["width"], 1, 1, 64, 1, 1, False
