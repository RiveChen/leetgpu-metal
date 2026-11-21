"""109. GRPO Surrogate Loss."""

import numpy as np


RTOL, ATOL = 0.0001, 0.0001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "groups": 2, "steps": 1, "clip": 0.2, "beta": 0.1},
    {"batch": 2, "groups": 4, "steps": 7, "clip": 0.15, "beta": 0.05},
    {"batch": 3, "groups": 7, "steps": 23, "clip": 0.25, "beta": 0.3},
]


def make_inputs(case, rng):
    batch = case["batch"]
    groups = case["groups"]
    steps = case["steps"]
    clip = case["clip"]
    beta = case["beta"]
    rewards = (
        rng.standard_normal((batch, groups), dtype=np.float32)
        if batch > 1
        else np.ones((batch, groups), np.float32)
    )
    logp = -np.abs(rng.standard_normal((batch, groups, steps), dtype=np.float32))
    old = -np.abs(rng.standard_normal((batch, groups, steps), dtype=np.float32))
    ref_logp = -np.abs(rng.standard_normal((batch, groups, steps), dtype=np.float32))
    return (
        [rewards, logp, old, ref_logp],
        (1,),
        [np.float32(clip), np.float32(beta), np.uint32(batch), np.uint32(groups), np.uint32(steps)],
    )


def reference_impl(rewards, logp, old, ref_logp, clip, beta, batch, groups, steps):
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    advantages = (centered / (rewards.std(axis=1, keepdims=True) + 1e-8))[..., None]
    ratio = np.exp(logp - old)
    policy = np.minimum(ratio * advantages, np.clip(ratio, 1 - clip, 1 + clip) * advantages)
    delta = ref_logp - logp
    return np.array([(beta * (np.expm1(delta) - delta) - policy).mean()], np.float32)


def dispatch_config(case):
    return 1, 1, 1, 64, 1, 1, False
