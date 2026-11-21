"""41. Simple Inference."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [
    {"batch": b, "features": f, "outputs": o, "bias": bias}
    for b, f, o, bias in ((1, 3, 2, True), (9, 7, 5, False), (257, 61, 33, True))
]


def make_inputs(case, rng):
    batch, features, outputs = case["batch"], case["features"], case["outputs"]
    x = rng.standard_normal((batch, features), dtype=np.float32)
    weights = rng.standard_normal((outputs, features), dtype=np.float32)
    bias = (
        rng.standard_normal(outputs, dtype=np.float32)
        if case["bias"]
        else np.zeros(outputs, np.float32)
    )
    return (
        [x, weights, bias],
        (batch, outputs),
        [np.uint32(batch), np.uint32(features), np.uint32(outputs)],
    )


def reference_impl(x, weights, bias):
    return x @ weights.T + bias


def dispatch_config(case):
    return case["batch"] * case["outputs"], 1, 1, 64, 1, 1, False
