"""147. Gated Attention."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"length": 1, "model": 3, "heads": 1},
    {"length": 11, "model": 12, "heads": 3},
    {"length": 27, "model": 32, "heads": 4},
]


def make_inputs(case, rng):
    length, model, heads = case["length"], case["model"], case["heads"]
    q, k, v, x = rng.standard_normal((4, length, model), dtype=np.float32)
    gate = rng.standard_normal((model, model), dtype=np.float32)
    return [q, k, v, x, gate], q.shape, [np.uint32(v) for v in (length, model, heads)]


def reference_impl(q, k, v, x, gate, length, model, heads):
    gate_logits = x.astype(np.float64) @ gate
    factors = np.exp(-np.logaddexp(0, -gate_logits))
    out = np.empty_like(q)
    causal = np.tri(length, dtype=bool)
    for idx in np.array_split(np.arange(model), heads):
        scores = q[:, idx].astype(np.float64) @ k[:, idx].T / np.sqrt(len(idx))
        out[:, idx] = (softmax(np.where(causal, scores, -np.inf)) @ v[:, idx]) * factors[:, idx]
    return out


def softmax(scores):
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=-1, keepdims=True)


def dispatch_config(case):
    return case["length"] * case["model"], 1, 1, 64, 1, 1, False
