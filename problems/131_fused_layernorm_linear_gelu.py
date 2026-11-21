"""131. Fused LayerNorm Linear GELU."""

import math

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [
    {"rows": 1, "features": 1, "outputs": 1, "eps": 0.001},
    {"rows": 9, "features": 7, "outputs": 5, "eps": 1e-05},
    {"rows": 31, "features": 29, "outputs": 37, "eps": 0.0001},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    rows = case["rows"]
    features = case["features"]
    outputs = case["outputs"]
    eps = case["eps"]
    x = rng.standard_normal((rows, features), dtype=np.float32)
    x[0] = 5
    gamma = rng.standard_normal(features, dtype=np.float32)
    beta = rng.standard_normal(features, dtype=np.float32)
    weights = rng.standard_normal((features, outputs), dtype=np.float32)
    bias = rng.standard_normal(outputs, dtype=np.float32)
    return (
        [x, gamma, beta, weights, bias],
        (rows, outputs),
        [np.uint32(rows), np.uint32(features), np.uint32(outputs), np.float32(eps)],
    )


def reference_impl(x, gamma, beta, weights, bias, rows, features, outputs, eps):
    centered = x - x.mean(axis=1, keepdims=True)
    normalized = centered / np.sqrt(np.mean(centered**2, axis=1, keepdims=True) + eps)
    projected = (normalized * gamma + beta) @ weights + bias
    cdf = np.array(
        [0.5 * math.erfc(-float(v) / math.sqrt(2)) for v in projected.flat], dtype=np.float32
    ).reshape(projected.shape)
    return cdf * projected


def dispatch_config(case):
    return case["rows"] * case["outputs"], 1, 1, 64, 1, 1, False
