"""93. Llama Transformer Block."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True

TEST_CASES = [{"seq_len": 1}, {"seq_len": 3}, {"seq_len": 7}]

WEIGHT_SHAPES = {
    "rms1": (512,),
    "q": (512, 512),
    "k": (128, 512),
    "v": (128, 512),
    "o": (512, 512),
    "rms2": (512,),
    "gate": (1408, 512),
    "up": (1408, 512),
    "down": (512, 1408),
}


def unpack(weights):
    tensors = {}
    offset = 0
    for name, shape in WEIGHT_SHAPES.items():
        count = int(np.prod(shape))
        tensors[name] = weights[offset : offset + count].reshape(shape)
        offset += count
    return tensors


def rms_norm(x, scale):
    return x / np.sqrt(np.mean(x**2, axis=1, keepdims=True) + 1e-5) * scale


def rotate(x, cosine, sine):
    lo, hi = np.split(x, 2, axis=-1)
    c, s = cosine[:, None, :], sine[:, None, :]
    return np.concatenate([lo * c - hi * s, lo * s + hi * c], axis=-1)


def make_inputs(case, rng):
    seq_len = case["seq_len"]
    shape = (seq_len, 512)
    x = rng.uniform(-1, 1, shape).astype(np.float32)
    weights = rng.standard_normal(
        sum(int(np.prod(s)) for s in WEIGHT_SHAPES.values()), dtype=np.float32
    ) * np.float32(0.025)
    p = unpack(weights)

    p["rms1"] += 1
    p["rms2"] += 1
    angle = np.arange(seq_len)[:, None] * 10000 ** (-np.arange(32) / 32)
    cosine = np.cos(angle).astype(np.float32)
    sine = np.sin(angle).astype(np.float32)
    return [x, weights, cosine, sine], shape, [np.uint32(seq_len)]


def reference_impl(x, weights, cosine, sine, seq_len):
    p = unpack(weights)

    norm = rms_norm(x, p["rms1"])
    q = rotate((norm @ p["q"].T).reshape(seq_len, 8, 64), cosine, sine)
    k = rotate((norm @ p["k"].T).reshape(seq_len, 2, 64), cosine, sine)
    v = (norm @ p["v"].T).reshape(seq_len, 2, 64)
    k = np.repeat(k, 4, axis=1)
    v = np.repeat(v, 4, axis=1)

    scores = np.einsum("thd,shd->hts", q, k) * 0.125
    scores[:, np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)] = -np.inf
    prob = np.exp(scores - scores.max(axis=-1, keepdims=True))
    context = np.einsum("hts,shd->thd", prob / prob.sum(axis=-1, keepdims=True), v).reshape(x.shape)
    hidden = x + context @ p["o"].T
    norm = rms_norm(hidden, p["rms2"])
    gate = norm @ p["gate"].T
    return hidden + ((gate / (1 + np.exp(-gate))) * (norm @ p["up"].T)) @ p["down"].T


def dispatch_config(case):
    return 1, 1, 1, 128, 1, 1, True


def workspace_bytes(case):
    return 4 * case["seq_len"] * 4608
