"""98. Beam Search Step."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "beams": 1, "vocab": 1},
    {"batch": 2, "beams": 3, "vocab": 7},
    {"batch": 3, "beams": 7, "vocab": 23},
]


def make_inputs(case, rng):
    batch = case["batch"]
    beams = case["beams"]
    vocab = case["vocab"]
    previous = -rng.integers(0, 4, (batch, beams)).astype(np.float32)
    logp = -rng.permutation(batch * beams * vocab).reshape(batch, beams, vocab).astype(
        np.float32
    ) / (batch * beams * vocab + 1)
    return (
        [previous, logp],
        [(previous.shape, np.float32), (previous.shape, np.int32), (previous.shape, np.int32)],
        [np.uint32(batch), np.uint32(beams), np.uint32(vocab)],
    )


def reference_impl(previous, logp, batch, beams, vocab):
    candidates = (previous[..., None] + logp).reshape(batch, -1)
    order = np.argsort(-candidates, axis=1)[:, :beams]
    return (
        np.take_along_axis(candidates, order, axis=1),
        (order // vocab).astype(np.int32),
        (order % vocab).astype(np.int32),
    )


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, False
