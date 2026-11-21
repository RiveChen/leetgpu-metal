"""107. PPO Clipped Surrogate Loss."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "steps": 1, "clip": 0.2},
    {"batch": 2, "steps": 9, "clip": 0.1},
    {"batch": 7, "steps": 37, "clip": 0.3},
]


def make_inputs(case, rng):
    batch = case["batch"]
    steps = case["steps"]
    clip = case["clip"]
    advantages = rng.standard_normal((batch, steps), dtype=np.float32)
    logp = -np.abs(rng.standard_normal((batch, steps), dtype=np.float32))
    old = -np.abs(rng.standard_normal((batch, steps), dtype=np.float32))
    return [advantages, logp, old], (1,), [np.float32(clip), np.uint32(batch), np.uint32(steps)]


def reference_impl(advantages, logp, old, clip, batch, steps):
    ratio = np.exp(logp - old)
    surrogate = np.minimum(ratio * advantages, np.clip(ratio, 1 - clip, 1 + clip) * advantages)
    return np.array([-surrogate.mean()], np.float32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
