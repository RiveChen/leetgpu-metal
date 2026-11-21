"""114. Multi-Head Latent Attention Decode."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True

TEST_CASES = [
    {"heads": 1, "length": 1, "rank": 2, "width": 3, "rope": 1},
    {"heads": 3, "length": 7, "rank": 5, "width": 4, "rope": 2},
    {"heads": 4, "length": 71, "rank": 13, "width": 9, "rope": 7},
]


def make_inputs(case, rng):
    heads = case["heads"]
    length = case["length"]
    rank = case["rank"]
    width = case["width"]
    rope = case["rope"]
    shape = (heads, width)
    q = rng.standard_normal((heads, width + rope), dtype=np.float32)
    cache = rng.standard_normal((length, rank + rope), dtype=np.float32)
    uk = rng.standard_normal((heads, width, rank), dtype=np.float32) / np.float32(np.sqrt(width))
    uv = rng.standard_normal((heads, rank, width), dtype=np.float32) / np.float32(np.sqrt(rank))
    return (
        [q, cache, uk, uv],
        shape,
        [np.uint32(heads), np.uint32(length), np.uint32(rank), np.uint32(width), np.uint32(rope)],
    )


def reference_impl(q, cache, uk, uv, heads, length, rank, width, rope):
    latent = cache[:, :rank]
    q_latent = np.einsum("hd,hdr->hr", q[:, :width], uk)
    scores = (q_latent @ latent.T + q[:, width:] @ cache[:, rank:].T) / np.float32(
        np.sqrt(width + rope)
    )
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    return np.einsum("hr,hrd->hd", (weights / weights.sum(axis=1, keepdims=True)) @ latent, uv)


def dispatch_config(case):
    return case["heads"], 1, 1, 64, 1, 1, True


def workspace_bytes(case):
    return 4 * case["heads"] * case["length"]
