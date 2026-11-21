"""26. Multi-Head Cross-Attention."""

import numpy as np

RTOL, ATOL = 1e-4, 1e-4
TEST_CASES = [
    {"m": m, "n": n, "heads": h, "width": d}
    for m, n, h, d in ((1, 4, 2, 3), (8, 1, 3, 5), (17, 23, 4, 7))
]


def make_inputs(case, rng):
    m, n, heads, width = (case[k] for k in ("m", "n", "heads", "width"))
    q, k, v = (rng.standard_normal((size, heads, width), dtype=np.float32) for size in (m, n, n))
    return [q, k, v], q.shape, [np.uint32(x) for x in (m, n, heads, width)]


def reference_impl(q, k, v):
    scores = np.einsum("mhd,nhd->hmn", q, k) / np.sqrt(q.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return np.einsum("hmn,nhd->mhd", weights / weights.sum(axis=-1, keepdims=True), v).astype(
        np.float32
    )


def dispatch_config(case):
    return case["m"] * case["heads"], 1, 1, 64, 1, 1, False
