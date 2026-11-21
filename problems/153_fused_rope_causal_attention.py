"""153. Fused RoPE Causal Attention."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"heads": 1, "kv": 1, "length": 1, "width": 2},
    {"heads": 4, "kv": 2, "length": 9, "width": 6},
    {"heads": 6, "kv": 2, "length": 19, "width": 16},
]


def make_inputs(case, rng):
    heads, kv, length, width = case["heads"], case["kv"], case["length"], case["width"]
    q = rng.standard_normal((heads, length, width), dtype=np.float32)
    k, v = rng.standard_normal((2, kv, length, width), dtype=np.float32)

    angle = rng.uniform(-np.pi, np.pi, (length, width // 2))
    cos = np.cos(angle).astype(np.float32)
    sin = np.sin(angle).astype(np.float32)
    return [q, k, v, cos, sin], q.shape, [np.uint32(v) for v in (heads, kv, length, width)]


def reference_impl(q, k, v, cos, sin, heads, kv, length, width):
    def rotate(x):
        a, b = np.split(x.astype(np.float64), 2, axis=-1)
        return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)

    queries, keys = rotate(q), rotate(k)
    out = np.empty_like(q)
    causal = np.tri(length, dtype=bool)
    for h in range(heads):
        group = h // (int(heads) // int(kv))
        scores = queries[h] @ keys[group].T / np.sqrt(float(width))
        out[h] = softmax(np.where(causal, scores, -np.inf)) @ v[group]
    return out


def softmax(scores):
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True)


def dispatch_config(case):
    return case["heads"] * case["length"] * case["width"], 1, 1, 64, 1, 1, False
