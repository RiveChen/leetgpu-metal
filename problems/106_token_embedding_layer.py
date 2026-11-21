"""106. Token Embedding Layer."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "length": 1, "vocab": 2, "positions": 3, "width": 1, "eps": 1e-05},
    {"batch": 2, "length": 5, "vocab": 11, "positions": 9, "width": 7, "eps": 1e-05},
    {"batch": 4, "length": 17, "vocab": 41, "positions": 23, "width": 15, "eps": 1e-05},
]


def make_inputs(case, rng):
    batch = case["batch"]
    length = case["length"]
    vocab = case["vocab"]
    positions = case["positions"]
    width = case["width"]
    eps = case["eps"]
    ids = rng.integers(0, vocab, (batch, length), dtype=np.int32)
    pos = rng.integers(0, positions, length, dtype=np.int32)
    word = rng.standard_normal((vocab, width), dtype=np.float32)
    position = rng.standard_normal((positions, width), dtype=np.float32)
    gamma = rng.standard_normal(width, dtype=np.float32)
    beta = rng.standard_normal(width, dtype=np.float32)
    return (
        [ids, pos, word, position, gamma, beta],
        (batch, length, width),
        [
            np.uint32(batch),
            np.uint32(length),
            np.uint32(vocab),
            np.uint32(positions),
            np.uint32(width),
            np.float32(eps),
        ],
    )


def reference_impl(
    ids, pos, word, position, gamma, beta, batch, length, vocab, positions, width, eps
):
    x = word[ids] + position[pos]
    centered = x - x.mean(axis=-1, keepdims=True)
    return centered / np.sqrt(x.var(axis=-1, keepdims=True) + eps) * gamma + beta


def dispatch_config(case):
    return case["batch"] * case["length"], 1, 1, 64, 1, 1, False
