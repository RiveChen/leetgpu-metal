"""155. MoE Expert Weight Gradient."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"experts": 1, "tokens": 1, "features": 1, "outputs": 1},
    {"experts": 5, "tokens": 3, "features": 7, "outputs": 2},
    {"experts": 7, "tokens": 43, "features": 17, "outputs": 13},
]


def make_inputs(case, rng):
    experts = case["experts"]
    tokens = case["tokens"]
    features = case["features"]
    outputs = case["outputs"]
    x = rng.standard_normal((tokens, features), dtype=np.float32)
    dy = rng.standard_normal((tokens, outputs), dtype=np.float32)
    cuts = np.sort(rng.integers(0, tokens + 1, max(0, experts - 1)))
    if experts > 1:
        cuts[0] = 0
    offsets = np.array([0, *cuts, tokens], dtype=np.int32)
    return (
        [x, dy, offsets],
        (experts, features, outputs),
        [np.uint32(v) for v in (experts, tokens, features, outputs)],
    )


def reference_impl(x, dy, offsets):
    return np.stack(
        [x[start:end].T @ dy[start:end] for start, end in zip(offsets[:-1], offsets[1:])]
    )


def dispatch_config(case):
    return case["experts"] * case["features"] * case["outputs"], 1, 1, 64, 1, 1, False
