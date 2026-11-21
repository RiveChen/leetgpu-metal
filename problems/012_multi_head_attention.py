"""12. Multi-Head Attention."""

import numpy as np

RTOL, ATOL = 1e-5, 1e-5
REFERENCE_USES_SCALARS = True
TEST_CASES = [{"n": n, "width": w, "heads": h} for n, w, h in ((1, 8, 2), (9, 15, 3), (33, 32, 4))]


def make_inputs(case, rng):
    n, width, heads = case["n"], case["width"], case["heads"]
    inputs = [rng.standard_normal((n, width), dtype=np.float32) for _ in range(3)]
    return inputs, inputs[0].shape, [np.uint32(v) for v in (n, width, heads)]


def reference_impl(q, k, v, n, width, heads):
    shape = (int(n), int(heads), int(width) // int(heads))
    q, k, v = (x.reshape(shape).transpose(1, 0, 2) for x in (q, k, v))
    scores = (q @ k.swapaxes(1, 2)) / np.sqrt(shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return (
        ((weights / weights.sum(axis=-1, keepdims=True)) @ v)
        .transpose(1, 0, 2)
        .reshape(n, width)
        .astype(np.float32)
    )


def dispatch_config(case):
    return case["n"] * case["heads"], 1, 1, 64, 1, 1, False
