"""130. Fused MoE SwiGLU Layer."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [
    {"tokens": 1, "hidden": 1, "inner": 3, "experts": 1, "top": 1},
    {"tokens": 5, "hidden": 13, "inner": 9, "experts": 4, "top": 2},
    {"tokens": 9, "hidden": 19, "inner": 73, "experts": 5, "top": 4},
]


def make_inputs(case, rng):
    tokens = case["tokens"]
    hidden = case["hidden"]
    inner = case["inner"]
    experts = case["experts"]
    top = case["top"]
    x = rng.standard_normal((tokens, hidden), dtype=np.float32)
    ids = np.stack([rng.choice(experts, top, replace=False) for _ in range(tokens)]).astype(
        np.int32
    )
    mix = rng.random((tokens, top), dtype=np.float32)
    mix /= mix.sum(axis=1, keepdims=True)

    def matrix(shape, fan):
        return rng.standard_normal(shape, dtype=np.float32) / np.sqrt(np.float32(fan))

    gate = matrix((experts, hidden, inner), hidden)
    up = matrix(gate.shape, hidden)
    down = matrix((experts, inner, hidden), inner)
    return (
        [x, ids, mix, gate, up, down],
        x.shape,
        [np.uint32(v) for v in (tokens, hidden, inner, experts, top)],
    )


def reference_impl(x, ids, mix, gate, up, down):
    out = np.zeros_like(x)
    for e in range(len(gate)):
        rows, slots = np.where(ids == e)
        activation = x[rows] @ gate[e]
        state = (activation / (1 + np.exp(-activation))) * (x[rows] @ up[e])
        out[rows] += mix[rows, slots, None] * (state @ down[e])
    return out


def dispatch_config(case):
    return 64 * case["tokens"], 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 4 * case["tokens"] * case["inner"]
