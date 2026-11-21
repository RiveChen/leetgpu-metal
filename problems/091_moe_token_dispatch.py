"""91. MoE Token Dispatch."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
REFERENCE_USES_SCALARS = True


TEST_CASES = [
    {"tokens": 1, "width": 3, "experts": 3, "capacity": 1},
    {"tokens": 9, "width": 5, "experts": 4, "capacity": 9},
    {"tokens": 67, "width": 11, "experts": 9, "capacity": 67},
]


def make_inputs(case, rng):
    tokens = case["tokens"]
    width = case["width"]
    experts = case["experts"]
    capacity = case["capacity"]
    x = rng.standard_normal((tokens, width), dtype=np.float32)
    routes = rng.permutation(np.arange(tokens, dtype=np.int32) % experts)
    return (
        [x, routes],
        [((experts, capacity, width), np.float32), ((experts,), np.int32)],
        [np.uint32(tokens), np.uint32(width), np.uint32(experts), np.uint32(capacity)],
    )


def reference_impl(x, routes, tokens, width, experts, capacity):
    result = np.zeros((experts, capacity, width), np.float32)
    counts = np.bincount(routes, minlength=experts).astype(np.int32)
    for expert, count in enumerate(counts):
        result[expert, :count] = x[routes == expert]
    return result, counts


def dispatch_config(case):
    return case["experts"], 1, 1, 64, 1, 1, False
