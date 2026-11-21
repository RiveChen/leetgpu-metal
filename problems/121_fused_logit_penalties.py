"""121. Fused Logit Penalties."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"batch": 1, "vocab": 4, "prompt_len": 2, "generated_len": 1},
    {"batch": 3, "vocab": 11, "prompt_len": 7, "generated_len": 9},
    {"batch": 7, "vocab": 61, "prompt_len": 19, "generated_len": 33},
]


def make_inputs(case, rng):
    batch = case["batch"]
    vocab = case["vocab"]
    prompt_len = case["prompt_len"]
    generated_len = case["generated_len"]
    logits = rng.standard_normal((batch, vocab), dtype=np.float32)
    prompt = rng.integers(-1, vocab, (batch, prompt_len), dtype=np.int32)
    generated = rng.integers(-1, vocab, (batch, generated_len), dtype=np.int32)
    generated[:, ::2] = 0
    presence = rng.uniform(-0.8, 1, batch).astype(np.float32)
    frequency = rng.uniform(-0.3, 1, batch).astype(np.float32)
    repetition = rng.uniform(0.5, 2, batch).astype(np.float32)
    return (
        [logits, prompt, generated, presence, frequency, repetition],
        logits.shape,
        [np.uint32(v) for v in (batch, vocab, prompt_len, generated_len)],
    )


def reference_impl(logits, prompt, generated, presence, frequency, repetition):
    result = logits.copy()
    for b, row in enumerate(result):
        counts = np.bincount(generated[b][generated[b] >= 0], minlength=row.size)
        seen = counts.astype(bool)
        seen[prompt[b][prompt[b] >= 0]] = True
        row[seen] = np.where(row[seen] > 0, row[seen] / repetition[b], row[seen] * repetition[b])
        row -= counts * frequency[b] + (counts > 0) * presence[b]
    return result


def dispatch_config(case):
    return case["batch"] * case["vocab"], 1, 1, 64, 1, 1, False
