"""126. Row-wise Argmax."""

import numpy as np

DTYPE = np.int32
RTOL, ATOL = 0.0, 0.0
TEST_CASES = [{"batch": 1, "vocab": 1}, {"batch": 9, "vocab": 17}, {"batch": 71, "vocab": 257}]


def make_inputs(case, rng):
    batch = case["batch"]
    vocab = case["vocab"]
    logits = rng.integers(-100, -1, (batch, vocab)).astype(np.float32)
    logits[::2, ::5] = 0
    return [logits], (batch,), [np.uint32(batch), np.uint32(vocab)]


def reference_impl(logits):
    return np.argmax(logits, axis=1).astype(np.int32)


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, False
