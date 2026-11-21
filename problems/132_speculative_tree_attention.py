"""132. Speculative Tree Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"prefix": 1, "nodes": 1, "heads": 1, "width": 2},
    {"prefix": 5, "nodes": 6, "heads": 3, "width": 7},
    {"prefix": 79, "nodes": 11, "heads": 2, "width": 17},
]


def make_inputs(case, rng):
    prefix = case["prefix"]
    nodes = case["nodes"]
    heads = case["heads"]
    width = case["width"]
    shape = (nodes, heads, width)

    def tensor(n):
        return rng.uniform(-6, 6, (n, heads, width)).astype(np.float32)

    parents = np.array(
        [-1 if i == 0 or i == 4 else (i - 1) // 2 for i in range(nodes)], dtype=np.int32
    )
    return (
        [tensor(nodes), tensor(prefix), tensor(prefix), tensor(nodes), tensor(nodes), parents],
        shape,
        [np.uint32(v) for v in (prefix, nodes, heads, width)],
    )


def reference_impl(q, cache_k, cache_v, tree_k, tree_v, parents):
    mask = np.eye(len(q), dtype=bool)
    for t, parent in enumerate(parents):
        if parent >= 0:
            mask[t] |= mask[parent]
    visible = np.concatenate([np.ones((len(q), len(cache_k)), dtype=bool), mask], axis=1)
    k = np.concatenate([cache_k, tree_k])
    v = np.concatenate([cache_v, tree_v])
    scores = np.einsum("thd,shd->ths", q.astype(np.float64), k) / np.sqrt(q.shape[-1])
    scores = np.where(visible[:, None, :], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return np.einsum("ths,shd->thd", weights / weights.sum(axis=-1, keepdims=True), v).astype(
        np.float32
    )


def dispatch_config(case):
    return case["nodes"] * case["heads"] * case["width"], 1, 1, 64, 1, 1, False
