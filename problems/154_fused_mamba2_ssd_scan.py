"""154. Fused Mamba-2 SSD Scan."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"batch": 1, "length": 1, "heads": 1, "width": 2, "groups": 1, "states": 1},
    {"batch": 2, "length": 11, "heads": 4, "width": 5, "groups": 2, "states": 7},
    {"batch": 2, "length": 67, "heads": 6, "width": 3, "groups": 3, "states": 9},
]


def make_inputs(case, rng):
    batch, length, heads, width, groups, states = (
        case["batch"],
        case["length"],
        case["heads"],
        case["width"],
        case["groups"],
        case["states"],
    )
    shape = (batch, length, heads, width)
    x, z = rng.standard_normal((2, *shape), dtype=np.float32)
    dt = rng.standard_normal(shape[:-1], dtype=np.float32)
    bias, skip = rng.standard_normal((2, heads), dtype=np.float32)
    decay = -rng.uniform(0.01, 1.0, heads).astype(np.float32)
    b, c = rng.standard_normal((2, batch, length, groups, states), dtype=np.float32)
    return (
        [x, dt, bias, decay, b, c, skip, z],
        shape,
        [np.uint32(v) for v in (batch, length, heads, width, groups, states)],
    )


def reference_impl(x, dt, bias, decay, b, c, skip, z, batch, length, heads, width, groups, states):
    state = np.zeros((batch, heads, width, states), np.float64)
    group = np.arange(heads) // (int(heads) // int(groups))
    step = np.logaddexp(0, dt.astype(np.float64) + bias)
    out = np.empty_like(x)
    for t in range(length):
        delta = step[:, t]
        state *= np.exp(delta * decay)[:, :, None, None]
        state += (delta[:, :, None] * x[:, t])[:, :, :, None] * b[:, t][:, group, None, :]
        projected = np.einsum("bhwn,bhn->bhw", state, c[:, t][:, group])
        gate = z[:, t] * np.exp(-np.logaddexp(0, -z[:, t].astype(np.float64)))
        out[:, t] = (projected + skip[:, None] * x[:, t]) * gate
    return out


def dispatch_config(case):
    return case["batch"] * case["heads"] * case["width"], 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 4 * case["batch"] * case["heads"] * case["width"] * case["states"]
