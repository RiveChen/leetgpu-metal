"""141. Flash Decoding Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"queries": 1, "length": 1, "heads": 2, "width": 1},
    {"queries": 2, "length": 31, "heads": 1, "width": 8},
    {"queries": 3, "length": 273, "heads": 3, "width": 17},
]


def make_inputs(case, rng):
    queries = case["queries"]
    length = case["length"]
    heads = case["heads"]
    width = case["width"]

    def tensor(rows):
        return rng.uniform(-6, 6, (rows, heads, width)).astype(np.float32)

    q = tensor(queries)
    return (
        [q, tensor(length), tensor(length)],
        q.shape,
        [np.uint32(v) for v in (queries, length, heads, width)],
    )


def reference_impl(q, k, v):
    scores = np.einsum("thd,shd->ths", q.astype(np.float64), k) / np.sqrt(q.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return np.einsum("ths,shd->thd", weights / weights.sum(axis=-1, keepdims=True), v).astype(
        np.float32
    )


def dispatch_config(case):
    return 64 * case["queries"] * case["heads"] * case["width"], 1, 1, 64, 1, 1, False
