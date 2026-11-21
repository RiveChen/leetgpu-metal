"""90. Causal Depthwise Conv1d."""

import numpy as np


RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "length": 1, "width": 3, "taps": 1},
    {"batch": 2, "length": 5, "width": 7, "taps": 4},
    {"batch": 3, "length": 33, "width": 13, "taps": 3},
]


def make_inputs(case, rng):
    batch = case["batch"]
    length = case["length"]
    width = case["width"]
    taps = case["taps"]
    x = rng.standard_normal((batch, length, width), dtype=np.float32)
    weight = rng.standard_normal((width, taps), dtype=np.float32)
    bias = rng.standard_normal(width, dtype=np.float32)
    return (
        [x, weight, bias],
        x.shape,
        [np.uint32(batch), np.uint32(length), np.uint32(width), np.uint32(taps)],
    )


def reference_impl(x, weight, bias, batch, length, width, taps):
    result = np.broadcast_to(bias, x.shape).copy()
    for lag in range(min(taps, length)):
        result[:, lag:] += x[:, : int(length) - lag] * weight[:, lag]
    return result


def dispatch_config(case):
    return case["batch"] * case["length"] * case["width"], 1, 1, 64, 1, 1, False
