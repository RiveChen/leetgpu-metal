"""120. Kimi Delta Attention."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [
    {"batch": 1, "length": 1, "width": 1},
    {"batch": 2, "length": 11, "width": 7},
    {"batch": 3, "length": 31, "width": 16},
]


def make_inputs(case, rng):
    batch = case["batch"]
    length = case["length"]
    width = case["width"]
    shape = (batch, length, width)
    q = rng.standard_normal(shape, dtype=np.float32)
    k = rng.standard_normal(shape, dtype=np.float32) / np.float32(np.sqrt(width))
    v = rng.standard_normal(shape, dtype=np.float32)
    alpha = rng.uniform(0.1, 1, shape).astype(np.float32)
    beta = rng.random((batch, length), dtype=np.float32)
    return [q, k, v, alpha, beta], shape, [np.uint32(v) for v in shape]


def reference_impl(q, k, v, alpha, beta):
    batch, length, width = q.shape
    output = np.empty_like(v)
    memory = np.zeros((batch, width, width), dtype=np.float32)
    for t in range(length):
        memory *= alpha[:, t, :, None]
        residual = v[:, t] - np.einsum("bi,bij->bj", k[:, t], memory)
        memory += k[:, t, :, None] * (beta[:, t, None] * residual)[:, None, :]
        output[:, t] = np.einsum("bi,bij->bj", q[:, t], memory)
    return output


def dispatch_config(case):
    return case["batch"] * case["width"], 1, 1, 64, 1, 1, False
