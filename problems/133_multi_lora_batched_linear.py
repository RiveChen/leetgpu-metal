"""133. Multi-LoRA Batched Linear."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [
    {"tokens": 1, "features": 1, "outputs": 1, "rank": 1, "adapters": 1, "scale": 0.33},
    {"tokens": 13, "features": 7, "outputs": 5, "rank": 3, "adapters": 2, "scale": 0.7},
    {"tokens": 29, "features": 23, "outputs": 13, "rank": 4, "adapters": 5, "scale": 0.5},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    tokens = case["tokens"]
    features = case["features"]
    outputs = case["outputs"]
    rank = case["rank"]
    adapters = case["adapters"]
    scale = case["scale"]
    x = rng.standard_normal((tokens, features), dtype=np.float32)
    weights = rng.standard_normal((features, outputs), dtype=np.float32)
    a = rng.standard_normal((adapters, features, rank), dtype=np.float32)
    b = rng.standard_normal((adapters, rank, outputs), dtype=np.float32)
    ids = rng.integers(-1, adapters, tokens, dtype=np.int32)
    ids[::4] = -1
    return (
        [x, weights, a, b, ids],
        (tokens, outputs),
        [np.uint32(v) for v in (tokens, features, outputs, rank, adapters)] + [np.float32(scale)],
    )


def reference_impl(x, weights, a, b, ids, tokens, features, outputs, rank, adapters, scale):
    result = x @ weights
    for i, adapter in enumerate(ids):
        if adapter >= 0:
            result[i] += scale * ((x[i] @ a[adapter]) @ b[adapter])
    return result


def dispatch_config(case):
    return case["tokens"] * case["outputs"], 1, 1, 64, 1, 1, False
