"""104. Min-P Sampling."""

import numpy as np


RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "vocab": 1, "threshold": 1.0},
    {"batch": 3, "vocab": 17, "threshold": 0.0},
    {"batch": 7, "vocab": 37, "threshold": 0.25},
]


def make_inputs(case, rng):
    batch = case["batch"]
    vocab = case["vocab"]
    threshold = case["threshold"]
    logits = rng.standard_normal((batch, vocab), dtype=np.float32)
    return [logits], logits.shape, [np.float32(threshold), np.uint32(batch), np.uint32(vocab)]


def reference_impl(logits, threshold, batch, vocab):
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    shifted[shifted < threshold] = 0
    return shifted / shifted.sum(axis=1, keepdims=True)


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, False
