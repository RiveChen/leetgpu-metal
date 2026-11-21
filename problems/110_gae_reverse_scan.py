"""110. Parallel Reverse Scan (GAE)."""

import numpy as np


RTOL, ATOL = 0.001, 0.001
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"batch": 1, "steps": 1, "gamma": 0.9, "lam": 0.5},
    {"batch": 3, "steps": 9, "gamma": 0.99, "lam": 0.0},
    {"batch": 5, "steps": 33, "gamma": 0.95, "lam": 0.83},
]


def make_inputs(case, rng):
    batch = case["batch"]
    steps = case["steps"]
    gamma = case["gamma"]
    lam = case["lam"]
    rewards = rng.standard_normal((batch, steps), dtype=np.float32)
    values = rng.standard_normal((batch, steps), dtype=np.float32)
    return (
        [rewards, values],
        values.shape,
        [np.float32(gamma), np.float32(lam), np.uint32(batch), np.uint32(steps)],
    )


def reference_impl(rewards, values, gamma, lam, batch, steps):
    delta = rewards - values
    delta[:, :-1] += gamma * values[:, 1:]
    result = delta.copy()
    for t in range(int(steps) - 2, -1, -1):
        result[:, t] += gamma * lam * result[:, t + 1]
    return result


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, False
