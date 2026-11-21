"""34. Logistic Regression."""

import numpy as np

RTOL, ATOL = 0.01, 0.01
TEST_CASES = [
    {"samples": 23, "features": 1},
    {"samples": 61, "features": 5},
    {"samples": 167, "features": 11},
]


def make_inputs(case, rng):
    samples, features = case["samples"], case["features"]
    x = rng.standard_normal((samples, features), dtype=np.float32)
    signal = x @ rng.uniform(-0.4, 0.4, features).astype(np.float32)
    y = rng.binomial(1, 1 / (1 + np.exp(-signal))).astype(np.float32)
    return [x, y], (features,), [np.uint32(samples), np.uint32(features)]


def reference_impl(x, y):
    x = x.astype(np.float64)
    beta = np.zeros(x.shape[1])
    for _ in range(100):
        logits = x @ beta
        p = np.exp(-np.logaddexp(0, -logits))
        gradient = x.T @ (p - y) + 1e-6 * beta
        hessian = (x.T * np.maximum(p * (1 - p), 1e-8)) @ x + 1e-6 * np.eye(x.shape[1])
        delta = np.linalg.solve(hessian, gradient)
        beta -= delta
        if np.linalg.norm(delta) < 1e-8:
            break
    return beta.astype(np.float32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 4 * case["features"] * (case["features"] + 1)
