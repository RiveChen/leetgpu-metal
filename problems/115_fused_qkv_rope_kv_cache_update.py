"""115. Fused QKV Projection with RoPE and KV Cache Update. Initial and updated KV caches use separate buffers."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"batch": 1, "model": 1, "q_heads": 1, "kv_heads": 1, "width": 2, "capacity": 1},
    {"batch": 2, "model": 7, "q_heads": 4, "kv_heads": 2, "width": 8, "capacity": 5},
    {"batch": 4, "model": 23, "q_heads": 6, "kv_heads": 2, "width": 14, "capacity": 19},
]


def make_inputs(case, rng):
    batch = case["batch"]
    model = case["model"]
    q_heads = case["q_heads"]
    kv_heads = case["kv_heads"]
    width = case["width"]
    capacity = case["capacity"]
    x = rng.standard_normal((batch, model), dtype=np.float32)
    weights = rng.standard_normal((model, (q_heads + 2 * kv_heads) * width), dtype=np.float32)
    angles = rng.standard_normal((capacity, width // 2), dtype=np.float32)
    trig = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)
    positions = rng.integers(0, capacity, batch, dtype=np.int32)
    shape = (batch, kv_heads, capacity, width)
    keys = rng.standard_normal(shape, dtype=np.float32)
    values = rng.standard_normal(shape, dtype=np.float32)
    return (
        [x, weights, trig, positions, keys, values],
        [((batch, q_heads, width), np.float32), (shape, np.float32), (shape, np.float32)],
        [np.uint32(v) for v in (batch, model, q_heads, kv_heads, width, capacity)],
    )


def reference_impl(x, weights, trig, positions, keys, values):
    batch, kv_heads, _, width = keys.shape
    qkv = (x @ weights).reshape(batch, -1, width)
    q, k, v = np.split(qkv, [qkv.shape[1] - 2 * kv_heads, qkv.shape[1] - kv_heads], axis=1)
    cosine, sine = np.split(trig[positions, None, :], 2, axis=-1)

    def rotate(features):
        low, high = np.split(features, 2, axis=-1)
        return np.concatenate([low * cosine - high * sine, high * cosine + low * sine], axis=-1)

    updated_keys, updated_values = keys.copy(), values.copy()
    updated_keys[np.arange(batch), :, positions, :] = rotate(k)
    updated_values[np.arange(batch), :, positions, :] = v
    return rotate(q), updated_keys, updated_values


def dispatch_config(case):
    return (
        case["batch"] * case["width"] * max(case["q_heads"], case["kv_heads"] * case["capacity"]),
        1,
        1,
        64,
        1,
        1,
        False,
    )
