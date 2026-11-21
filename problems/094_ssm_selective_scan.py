"""94. SSM Selective Scan."""

import numpy as np


RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "length": 1, "width": 2, "states": 1},
    {"batch": 2, "length": 7, "width": 3, "states": 5},
    {"batch": 3, "length": 27, "width": 11, "states": 7},
]


def make_inputs(case, rng):
    batch = case["batch"]
    length = case["length"]
    width = case["width"]
    states = case["states"]
    u = rng.standard_normal((batch, length, width), dtype=np.float32)
    delta = rng.uniform(0.02, 1, u.shape).astype(np.float32)
    a = -rng.uniform(0.1, 1, (width, states)).astype(np.float32)
    b = rng.standard_normal((batch, length, states), dtype=np.float32)
    c = rng.standard_normal((batch, length, states), dtype=np.float32)
    skip = rng.standard_normal(width, dtype=np.float32)
    return (
        [u, delta, a, b, c, skip],
        u.shape,
        [np.uint32(batch), np.uint32(length), np.uint32(width), np.uint32(states)],
    )


def reference_impl(u, delta, a, b, c, skip, batch, length, width, states):
    result = np.empty_like(u)
    state = np.zeros((batch, width, states), np.float32)
    for step in range(length):
        dt = delta[:, step, :, None]
        state = np.exp(dt * a) * state + dt * u[:, step, :, None] * b[:, step, None, :]
        result[:, step] = np.einsum("bds,bs->bd", state, c[:, step]) + skip * u[:, step]
    return result


def dispatch_config(case):
    return case["batch"] * case["width"], 1, 1, 64, 1, 1, False
