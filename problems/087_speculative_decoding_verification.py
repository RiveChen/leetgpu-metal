"""87. Speculative Decoding Verification."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
DTYPE = np.int32
TEST_CASES = [
    {"batch": 1, "steps": 1, "vocab": 3},
    {"batch": 5, "steps": 3, "vocab": 7},
    {"batch": 17, "steps": 9, "vocab": 13},
]


def make_inputs(case, rng):
    batch, steps, vocab = case["batch"], case["steps"], case["vocab"]
    draft = rng.uniform(0.01, 1, (batch, steps, vocab)).astype(np.float32)
    draft /= draft.sum(axis=-1, keepdims=True)
    target = rng.uniform(0.01, 1, draft.shape).astype(np.float32)
    target /= target.sum(axis=-1, keepdims=True)
    tokens = rng.integers(0, vocab, (batch, steps), dtype=np.int32)
    samples = rng.random((batch, steps + 1), dtype=np.float32)
    samples[0, :steps] = 0
    if batch > 1:
        tokens[1, 0] = np.argmax(draft[1, 0] - target[1, 0])
        samples[1, 0] = np.nextafter(np.float32(1), np.float32(0))
    return (
        [tokens, draft, target, samples],
        samples.shape,
        [np.uint32(v) for v in (batch, steps, vocab)],
    )


def reference_impl(tokens, draft, target, samples):
    out = np.zeros(samples.shape, dtype=np.int32)
    for b, row in enumerate(tokens):
        for t, token in enumerate(row):
            if samples[b, t] < min(1, target[b, t, token] / draft[b, t, token]):
                out[b, t] = token
            else:
                distribution = np.maximum(target[b, t] - draft[b, t], 0)
                if not distribution.any():
                    distribution.fill(1)
                break
        else:
            distribution = target[b, -1]
            t = len(row)
        cdf = np.cumsum(distribution / distribution.sum())
        out[b, t] = min(np.searchsorted(cdf, samples[b, -1]), len(cdf) - 1)
    return out


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, False
