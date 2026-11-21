"""25. Categorical Cross Entropy Loss."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"samples": 1, "classes": 1},
    {"samples": 7, "classes": 13},
    {"samples": 149, "classes": 31},
]


def make_inputs(case, rng):
    samples, classes = case["samples"], case["classes"]
    logits = rng.standard_normal((samples, classes), dtype=np.float32)
    logits += rng.uniform(-1000, 1000, (samples, 1)).astype(np.float32)
    labels = rng.integers(0, classes, samples, dtype=np.int32)
    return [logits, labels], (1,), [np.uint32(samples), np.uint32(classes)]


def reference_impl(logits, labels):
    centered = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    losses = np.log(np.exp(centered).sum(axis=1)) - centered[np.arange(len(labels)), labels]
    return np.array([losses.mean()], dtype=np.float32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
