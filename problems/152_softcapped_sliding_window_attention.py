"""152. Softcapped Sliding Window Attention."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"heads": 1, "kv": 1, "length": 1, "width": 2, "window": 1, "cap": 0.5},
    {"heads": 4, "kv": 2, "length": 9, "width": 6, "window": 3, "cap": 0.8},
    {"heads": 6, "kv": 2, "length": 19, "width": 16, "window": 25, "cap": 4},
]


def make_inputs(case, rng):
    heads, kv, length, width, window, cap = (
        case["heads"],
        case["kv"],
        case["length"],
        case["width"],
        case["window"],
        case["cap"],
    )
    q = rng.standard_normal((heads, length, width), dtype=np.float32)
    k, v = rng.standard_normal((2, kv, length, width), dtype=np.float32)

    return (
        [3 * q, k, v],
        q.shape,
        [np.uint32(v) for v in (heads, kv, length, width, window)] + [np.float32(cap)],
    )


def reference_impl(q, k, v, heads, kv, length, width, window, cap):
    result = np.empty_like(q)
    pos = np.arange(length)
    visible = (pos[:, None] >= pos) & (pos[:, None] - pos < int(window))
    for h in range(heads):
        group = h // (int(heads) // int(kv))
        raw = q[h].astype(np.float64) @ k[group].T / np.sqrt(float(width))
        scores = cap * np.tanh(raw / cap)
        result[h] = softmax(np.where(visible, scores, -np.inf)) @ v[group]
    return result


def softmax(scores):
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True)


def dispatch_config(case):
    return case["heads"] * case["length"] * case["width"], 1, 1, 64, 1, 1, False
