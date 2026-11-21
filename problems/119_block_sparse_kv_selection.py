"""119. Block-Sparse KV Selection Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"heads": 1, "length": 1, "width": 1, "block": 3, "selected": 4},
    {"heads": 2, "length": 9, "width": 3, "block": 2, "selected": 2},
    {"heads": 3, "length": 139, "width": 13, "block": 7, "selected": 5},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    heads = case["heads"]
    length = case["length"]
    width = case["width"]
    block = case["block"]
    selected = case["selected"]
    q = rng.standard_normal((heads, width), dtype=np.float32)
    k = rng.standard_normal((heads, length, width), dtype=np.float32)
    v = rng.standard_normal(k.shape, dtype=np.float32)
    if heads > 1:
        k[0] = 1  # All block means tie for this head.
    return [q, k, v], q.shape, [np.uint32(v) for v in (heads, length, width, block, selected)]


def reference_impl(q, k, v, heads, length, width, block, selected):
    means = np.stack([k[:, s : s + block].mean(axis=1) for s in range(0, length, block)], axis=1)
    order = np.argsort(-np.einsum("hd,hbd->hb", q, means), axis=1, kind="stable")
    out = np.empty_like(q)
    for h in range(heads):
        mask = np.isin(np.arange(length) // block, order[h, :selected])
        logits = k[h, mask] @ q[h] / np.sqrt(np.float32(width))
        prob = np.exp(logits - logits.max())
        out[h] = prob @ v[h, mask] / prob.sum()
    return out


def dispatch_config(case):
    return 64 * case["heads"], 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 8 * case["heads"] * ((case["length"] + case["block"] - 1) // case["block"])
