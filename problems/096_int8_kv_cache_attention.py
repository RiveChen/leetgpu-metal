"""96. INT8 KV-Cache Attention."""

import numpy as np


RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"heads": 1, "length": 1, "width": 8},
    {"heads": 2, "length": 9, "width": 16},
    {"heads": 4, "length": 35, "width": 24},
]


def make_inputs(case, rng):
    heads = case["heads"]
    length = case["length"]
    width = case["width"]
    q = rng.standard_normal((heads, width), dtype=np.float32)
    keys = rng.integers(-128, 128, (heads, length, width), dtype=np.int8)
    values = rng.integers(-128, 128, keys.shape, dtype=np.int8)
    key_scale = rng.uniform(0.01, 0.05, (heads, length)).astype(np.float32)
    value_scale = rng.uniform(0.01, 0.05, (heads, length)).astype(np.float32)
    return (
        [q, keys, values, key_scale, value_scale],
        q.shape,
        [np.uint32(heads), np.uint32(length), np.uint32(width)],
    )


def reference_impl(q, keys, values, key_scale, value_scale, heads, length, width):
    keys_f = keys.astype(np.float32) * key_scale[..., None]
    values_f = values.astype(np.float32) * value_scale[..., None]
    scores = np.einsum("hd,htd->ht", q, keys_f) / np.float32(np.sqrt(width))
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return np.einsum("ht,htd->hd", weights / weights.sum(axis=1, keepdims=True), values_f)


def dispatch_config(case):
    return case["heads"] * case["width"], 1, 1, 64, 1, 1, False
