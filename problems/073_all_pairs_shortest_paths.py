"""73. All-Pairs Shortest Paths."""

import numpy as np

RTOL, ATOL = 0.01, 0.01
REFERENCE_USES_SCALARS = True
ALLOW_EXPECTED_INF = True

TEST_CASES = [{"nodes": 1}, {"nodes": 7}, {"nodes": 29}]


def make_inputs(case, rng):
    nodes = case["nodes"]
    shape = (nodes, nodes)
    # Positive base costs plus vertex potentials allow negative edges without negative cycles.
    potential = rng.integers(-9, 10, nodes)
    dist = (rng.integers(1, 11, shape) + potential[:, None] - potential).astype(np.float32)
    dist[rng.random(shape) < 0.7] = np.inf
    dist[-1] = np.inf
    np.fill_diagonal(dist, 0)
    return [dist], shape, [np.uint32(nodes)]


def reference_impl(dist, nodes):
    result = dist.copy()
    for k in range(nodes):
        result = np.minimum(result, result[:, k, None] + result[None, k, :])
    return result


def dispatch_config(case):
    return 1, 1, 1, 128, 1, 1, True
