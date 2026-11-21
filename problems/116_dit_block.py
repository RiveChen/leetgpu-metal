"""116. Diffusion Transformer Block."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [{"batch": 1, "length": 1}, {"batch": 1, "length": 5}, {"batch": 2, "length": 3}]


def make_inputs(case, rng):
    batch = case["batch"]
    length = case["length"]
    width = 512
    x = rng.uniform(-1, 1, (batch, length, width)).astype(np.float32)
    c = rng.uniform(-2, 2, (batch, width)).astype(np.float32)
    packed = []
    for rows, cols in [
        (6 * width, width),
        (3 * width, width),
        (width, width),
        (4 * width, width),
        (width, 4 * width),
    ]:
        w = rng.standard_normal((rows, cols), dtype=np.float32) / np.sqrt(np.float32(cols))
        bias = rng.uniform(-0.1, 0.1, rows).astype(np.float32)
        packed.extend((w.ravel(), bias))
    return [x, c, np.concatenate(packed)], x.shape, [np.uint32(batch), np.uint32(length)]


def reference_impl(x, c, weights):
    batch, length, width = x.shape
    pos = 0

    def linear(rows, cols):
        nonlocal pos
        w = weights[pos : pos + rows * cols].reshape(rows, cols)
        pos += rows * cols
        b = weights[pos : pos + rows]
        pos += rows
        return lambda z: z @ w.T + b

    def normalize(z):
        centered = z - z.mean(axis=-1, keepdims=True)
        return centered / np.sqrt((centered**2).mean(axis=-1, keepdims=True) + 1e-6)

    mod = linear(6 * width, width)(c / (1 + np.exp(-c))).reshape(batch, 6, width)
    qkv = linear(3 * width, width)(normalize(x) * (1 + mod[:, 1:2]) + mod[:, 0:1])
    q, k, v = np.split(qkv.reshape(batch, length, 3, 8, 64), 3, axis=2)
    scores = np.einsum("bthd,bshd->bhts", q[:, :, 0], k[:, :, 0]) / 8
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn = np.einsum("bhts,bshd->bthd", probs / probs.sum(axis=-1, keepdims=True), v[:, :, 0])
    residual = x + mod[:, 2:3] * linear(width, width)(attn.reshape(x.shape))
    hidden = linear(4 * width, width)(normalize(residual) * (1 + mod[:, 4:5]) + mod[:, 3:4])
    gelu = 0.5 * hidden * (1 + np.tanh(np.sqrt(2 / np.pi) * (hidden + 0.044715 * hidden**3)))
    return (residual + mod[:, 5:6] * linear(width, 4 * width)(gelu)).astype(np.float32)


def dispatch_config(case):
    return 64 * case["batch"], 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 4 * case["batch"] * 512 * (6 + 10 * case["length"])
