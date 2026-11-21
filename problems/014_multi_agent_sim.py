"""14. Multi-Agent Simulation."""

import numpy as np

RTOL, ATOL = 1e-5, 1e-5
TEST_CASES = [{"n": n} for n in (1, 5, 129)]


def make_inputs(case, rng):
    n = case["n"]
    agents = rng.uniform(-6, 6, (n, 4)).astype(np.float32)
    if n >= 5:
        # Include coincident positions and positions exactly at the exclusion radius.
        agents[:5, :2] = [[0, 0], [0, 0], [3, 4], [2, 1], [20, 20]]
    return [agents], agents.shape, [np.uint32(n)]


def reference_impl(agents):
    pos, velocity = agents[:, :2], agents[:, 2:]
    delta = pos[:, None, :] - pos[None, :, :]
    neighbors = (delta**2).sum(axis=-1) < 25
    np.fill_diagonal(neighbors, False)
    count = neighbors.sum(axis=1, keepdims=True)
    average = np.divide(neighbors @ velocity, count, out=velocity.copy(), where=count > 0)
    updated = velocity + 0.05 * (average - velocity)
    return np.column_stack((pos + updated, updated))


def dispatch_config(case):
    return case["n"], 1, 1, 64, 1, 1, False
