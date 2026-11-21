"""85. LoRA Linear."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"batch": 1, "inputs": 3, "outputs": 5, "rank": 1, "scale": 0.0},
    {"batch": 5, "inputs": 7, "outputs": 3, "rank": 2, "scale": 0.7},
    {"batch": 37, "inputs": 19, "outputs": 23, "rank": 5, "scale": -0.2},
]


def make_inputs(case, rng):
    batch, inputs, outputs, rank, scale = (
        case["batch"],
        case["inputs"],
        case["outputs"],
        case["rank"],
        case["scale"],
    )
    x = rng.standard_normal((batch, inputs), dtype=np.float32)
    w = rng.standard_normal((outputs, inputs), dtype=np.float32)
    a = rng.standard_normal((rank, inputs), dtype=np.float32)
    b = rng.standard_normal((outputs, rank), dtype=np.float32)
    return (
        [x, w, a, b],
        (batch, outputs),
        [np.uint32(v) for v in (batch, inputs, outputs, rank)] + [np.float32(scale)],
    )


def reference_impl(x, w, a, b, batch, inputs, outputs, rank, scale):
    return x @ w.T + scale * ((x @ a.T) @ b.T)


def dispatch_config(case):
    return case["batch"] * case["outputs"], 1, 1, 64, 1, 1, False
