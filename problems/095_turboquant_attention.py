"""95. TurboQuant KV Cache Attention."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True

TEST_CASES = [
    {"batch": 1, "length": 1, "width": 2, "levels": 4},
    {"batch": 2, "length": 7, "width": 9, "levels": 16},
    {"batch": 4, "length": 67, "width": 24, "levels": 256},
]


def make_inputs(case, rng):
    batch = case["batch"]
    length = case["length"]
    width = case["width"]
    levels = case["levels"]
    shape = (batch, length)
    q = rng.standard_normal((batch, width), dtype=np.float32)
    rotation = np.linalg.qr(rng.standard_normal((width, width)))[0].astype(np.float32)
    projection = rng.standard_normal((width, width), dtype=np.float32)
    book = np.linspace(-1, 1, levels, dtype=np.float32)
    idx = rng.integers(0, levels, (length, width), dtype=np.uint8)
    signs = rng.choice([-1, 1], size=idx.shape).astype(np.int8)
    gamma = rng.uniform(0, 2, length).astype(np.float32)
    gamma[0] = 0
    return (
        [q, idx, signs, gamma, rotation, projection, book],
        shape,
        [np.uint32(batch), np.uint32(length), np.uint32(width), np.uint32(levels)],
    )


def reference_impl(q, idx, signs, gamma, rotation, projection, book, batch, length, width, levels):
    base = book[idx] @ rotation
    residual = (signs.astype(np.float32) @ projection) * (
        gamma[:, None] * np.float32(np.sqrt(np.pi / 2) / width)
    )
    return q @ (base + residual).T


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, True
