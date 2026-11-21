"""101. N-body Gravitational Force."""

import numpy as np

RTOL, ATOL = 0.01, 0.01
REFERENCE_USES_SCALARS = True


TEST_CASES = [{"particles": 1}, {"particles": 9}, {"particles": 131}]


def make_inputs(case, rng):
    particles = case["particles"]
    positions = rng.standard_normal((particles, 3), dtype=np.float32)
    masses = rng.uniform(0.25, 4, particles).astype(np.float32)
    return [positions, masses], positions.shape, [np.uint32(particles)]


def reference_impl(positions, masses, particles):
    points = positions.astype(np.float64)
    result = np.empty_like(positions)
    for start in range(0, len(points), 64):
        targets = points[start : start + 64]
        offset = points[None, :, :] - targets[:, None, :]
        distance_squared = np.sum(offset**2, axis=-1) + 1e-6
        result[start : start + 64] = np.einsum(
            "ijc,ij->ic", offset, masses[None, :] / distance_squared**1.5
        )
    return result


def dispatch_config(case):
    return case["particles"], 1, 1, 64, 1, 1, False
