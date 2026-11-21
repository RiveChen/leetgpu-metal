"""86. Paged KV-Cache Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001

TEST_CASES = [
    {"batch": 1, "heads": 1, "width": 3, "block": 1, "pages": 1},
    {"batch": 3, "heads": 4, "width": 7, "block": 4, "pages": 3},
    {"batch": 7, "heads": 3, "width": 9, "block": 7, "pages": 5},
]


def make_inputs(case, rng):
    batch, heads, width, block, pages = (
        case["batch"],
        case["heads"],
        case["width"],
        case["block"],
        case["pages"],
    )
    q = rng.standard_normal((batch, heads, width), dtype=np.float32)
    k = rng.standard_normal((batch * pages, block, heads, width), dtype=np.float32)
    v = rng.standard_normal(k.shape, dtype=np.float32)
    table = rng.permutation(batch * pages).reshape(batch, pages).astype(np.int32)
    lengths = rng.integers(1, pages * block + 1, batch, dtype=np.int32)
    lengths[-1] = pages * block
    return (
        [q, k, v, table, lengths],
        q.shape,
        [np.uint32(v) for v in (batch, heads, width, block, pages)],
    )


def reference_impl(q, k, v, table, lengths):
    out = np.empty_like(q)
    for b in range(len(q)):
        positions = np.arange(lengths[b])
        physical = table[b, positions // k.shape[1]]
        indices = (physical, positions % k.shape[1])
        scores = np.einsum("hd,thd->th", q[b], k[indices]) / np.float32(np.sqrt(q.shape[-1]))
        weights = np.exp(scores - scores.max(axis=0, keepdims=True))
        out[b] = np.einsum("th,thd->hd", weights / weights.sum(axis=0), v[indices])
    return out


def dispatch_config(case):
    return case["batch"] * case["heads"] * case["width"], 1, 1, 64, 1, 1, False
