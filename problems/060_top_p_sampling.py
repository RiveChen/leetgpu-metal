"""60. Top-p Sampling."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
DTYPE = np.int32
TEST_CASES = [
    {"n": 9, "threshold": 0.02, "seed": 34},
    {"n": 37, "threshold": 0.68, "seed": 398},
    {"n": 81, "threshold": 1.0, "seed": -5},
]


def pcg(x):
    state = (x * 747796405 + 2891336453) & 0xFFFFFFFF
    word = (((state >> ((state >> 28) + 4)) ^ state) * 277803737) & 0xFFFFFFFF
    return word ^ (word >> 22)


def make_inputs(case, rng):
    n, threshold, seed = case["n"], case["threshold"], case["seed"]
    logits = rng.standard_normal(n, dtype=np.float32)
    logits[:2] = 2.5
    return (
        [logits, np.array([threshold], np.float32), np.array([seed], np.int32)],
        (1,),
        [np.uint32(n)],
    )


def reference_impl(logits, p, seed):
    weights = np.exp(logits - logits.max())
    order = np.argsort(-weights, kind="stable")
    cumulative = np.cumsum(weights[order] / weights.sum())
    count = min(len(order), int(np.searchsorted(cumulative, p[0])) + 1)
    nucleus = order[:count]
    key = pcg(int(seed[0]))
    u = np.array([(pcg(key + int(token)) + 0.5) / 2**32 for token in nucleus])
    return np.array([nucleus[np.argmax(weights[nucleus] / -np.log(u))]], np.int32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
