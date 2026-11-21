"""148. Differential Attention."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"length": 1, "heads": 1, "width": 1, "initial": 0.2},
    {"length": 9, "heads": 3, "width": 5, "initial": 0.4},
    {"length": 29, "heads": 4, "width": 8, "initial": 0.7},
]


def make_inputs(case, rng):
    length, heads, width, initial = case["length"], case["heads"], case["width"], case["initial"]
    q, k, v = rng.standard_normal((3, length, heads, 2 * width), dtype=np.float32)
    lam = rng.uniform(0.1, 1.2, heads).astype(np.float32)
    lam[0] = 1
    return (
        [q, k, v, lam],
        q.shape,
        [np.uint32(v) for v in (length, heads, width)] + [np.float32(initial)],
    )


def reference_impl(q, k, v, lam, length, heads, width, initial):
    out = np.empty_like(q)
    mask = np.tri(length, dtype=bool)
    for h in range(heads):
        maps = []
        for qs, ks in zip(
            np.split(q[:, h], 2, axis=-1), np.split(k[:, h], 2, axis=-1), strict=True
        ):
            scores = qs.astype(np.float64) @ ks.T / np.sqrt(float(width))
            maps.append(softmax(np.where(mask, scores, -np.inf)))
        diff = (maps[0] - lam[h] * maps[1]) @ v[:, h]
        out[:, h] = (1 - initial) * diff / np.sqrt(np.mean(diff**2, axis=1, keepdims=True) + 1e-5)
    return out


def softmax(scores):
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True)


def dispatch_config(case):
    return case["length"] * case["heads"], 1, 1, 64, 1, 1, False
