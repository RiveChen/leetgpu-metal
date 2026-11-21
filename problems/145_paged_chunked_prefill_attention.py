"""145. Paged Chunked Prefill Attention."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"contexts": [1], "chunks": [1], "heads": 1, "kv": 1, "width": 3, "block": 4},
    {"contexts": [7, 3], "chunks": [3, 3], "heads": 4, "kv": 2, "width": 7, "block": 2},
    {"contexts": [9, 19, 2], "chunks": [1, 8, 2], "heads": 6, "kv": 2, "width": 13, "block": 4},
]


def make_inputs(case, rng):
    contexts, chunks, heads, kv, width, block = (
        case["contexts"],
        case["chunks"],
        case["heads"],
        case["kv"],
        case["width"],
        case["block"],
    )
    lengths = np.array(contexts, np.int32)
    offsets = np.array([0, *np.cumsum(chunks)], np.int32)
    pages = (lengths + block - 1) // block
    slots = int(pages.max())
    order = rng.permutation(int(sum(pages)) + 2).astype(np.int32)
    table = np.full((len(lengths), slots), -1, np.int32)
    cursor = 0
    for seq, count in enumerate(pages):
        table[seq, :count] = order[cursor : cursor + count]
        cursor += count
    q = rng.standard_normal((sum(chunks), heads, width), dtype=np.float32)
    k, v = rng.standard_normal((2, len(order), block, kv, width), dtype=np.float32)
    return (
        [q, k, v, table, lengths, offsets],
        q.shape,
        [np.uint32(v) for v in (len(chunks), sum(chunks), heads, kv, width, block, slots)],
    )


def reference_impl(q, k, v, table, lengths, offsets, seqs, total, heads, kv, width, block, slots):
    out = np.empty_like(q)
    for s in range(seqs):
        start, end = offsets[s : s + 2]
        positions = np.arange(lengths[s])
        pages = table[s, positions // int(block)]
        keys = k[pages, positions % int(block)].astype(np.float64)
        values = v[pages, positions % int(block)]
        allowed = positions <= np.arange(lengths[s] - (end - start), lengths[s])[:, None]
        for h in range(heads):
            group = h // (int(heads) // int(kv))
            scores = q[start:end, h] @ keys[:, group].T / np.sqrt(float(width))
            out[start:end, h] = softmax(np.where(allowed, scores, -np.inf)) @ values[:, group]
    return out


def softmax(scores):
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True)


def dispatch_config(case):
    return sum(case["chunks"]) * case["heads"] * case["width"], 1, 1, 64, 1, 1, False
