"""20. K-Means Clustering."""

import numpy as np

RTOL, ATOL = 1e-4, 1e-4
REFERENCE_USES_SCALARS = True
TEST_CASES = [
    {"n": n, "k": k, "steps": steps} for n, k, steps in ((11, 3, 1), (139, 6, 7), (3, 4, 3))
]


def make_inputs(case, rng):
    n, k, steps = case["n"], case["k"], case["steps"]
    x, y = rng.standard_normal((2, n), dtype=np.float32)
    cx, cy = rng.standard_normal((2, k), dtype=np.float32)
    if n == 3:
        # The duplicate initial centroid tests first-index ties and empty clusters.
        x[:], y[:] = [-1, 0, 1], [0, 0, 0]
        cx[:], cy[:] = [-1, -1, 1, 20], [0, 0, 0, 20]
    outputs = [((n,), np.int32), ((k,), np.float32), ((k,), np.float32)]
    return [x, y, cx, cy], outputs, [np.uint32(z) for z in (n, k, steps)]


def reference_impl(x, y, cx, cy, n, k, steps):
    centers = np.column_stack((cx, cy))
    points = np.column_stack((x, y))
    for _ in range(steps):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        labels = d2.argmin(axis=1).astype(np.int32)
        for cluster in range(k):
            members = points[labels == cluster]
            if members.size:
                centers[cluster] = members.mean(axis=0)
    return labels, centers[:, 0], centers[:, 1]


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
