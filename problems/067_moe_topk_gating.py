"""67. MoE Top-K Gating."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True
TEST_CASES = [{"m": 1, "e": 3, "k": 1}, {"m": 5, "e": 8, "k": 8}, {"m": 17, "e": 31, "k": 4}]


def make_inputs(case, rng):
    m, e, k = case["m"], case["e"], case["k"]
    logits = np.stack([rng.permutation(e) for _ in range(m)]).astype(np.float32) / 4
    return (
        [logits],
        [((m, k), np.float32), ((m, k), np.int32)],
        [np.uint32(v) for v in (m, e, k)],
    )


def reference_impl(logits, m, e, k):
    order = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, order, axis=1)
    weights = np.exp(top - top.max(axis=1, keepdims=True))
    return weights / weights.sum(axis=1, keepdims=True), order.astype(np.int32)


def dispatch_config(case):
    return case["m"], 1, 1, 64, 1, 1, False
