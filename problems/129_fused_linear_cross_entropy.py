"""129. Fused Linear Cross Entropy."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [
    {"rows": 1, "width": 1, "vocab": 2},
    {"rows": 7, "width": 9, "vocab": 23},
    {"rows": 67, "width": 17, "vocab": 71},
]


def make_inputs(case, rng):
    rows = case["rows"]
    width = case["width"]
    vocab = case["vocab"]
    x = rng.standard_normal((rows, width), dtype=np.float32)
    w = rng.standard_normal((vocab, width), dtype=np.float32)
    if rows > 1:
        x[0] *= 50
    targets = rng.integers(0, vocab, rows, dtype=np.int32)
    return [x, w, targets], (1,), [np.uint32(v) for v in (rows, width, vocab)]


def reference_impl(x, w, targets):
    z = x.astype(np.float64) @ w.T
    peak = z.max(axis=1)
    losses = (peak - z[np.arange(len(x)), targets]) + np.log(np.exp(z - peak[:, None]).sum(axis=1))
    return np.array([losses.mean()], dtype=np.float32)


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
