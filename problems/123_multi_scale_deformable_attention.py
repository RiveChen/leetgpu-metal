"""123. Multi-Scale Deformable Attention."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"queries": 1, "heads": 1, "width": 1, "levels": 1, "points": 1},
    {"queries": 3, "heads": 2, "width": 7, "levels": 2, "points": 4},
    {"queries": 9, "heads": 3, "width": 5, "levels": 4, "points": 3},
]


def make_inputs(case, rng):
    queries = case["queries"]
    heads = case["heads"]
    width = case["width"]
    levels = case["levels"]
    points = case["points"]
    shapes = np.array(
        [(max(1, 6 // (j + 1)), max(1, 9 // (j + 1))) for j in range(levels)], dtype=np.int32
    )
    value = rng.standard_normal((int(shapes.prod(axis=1).sum()), heads, width), dtype=np.float32)
    locations = rng.uniform(-0.25, 1.25, (queries, heads, levels, points, 2)).astype(np.float32)
    locations[0, 0, 0, 0] = 0
    weights = rng.uniform(-5, 5, locations.shape[:-1]).astype(np.float32)
    return (
        [value, shapes, locations, weights],
        (queries, heads, width),
        [np.uint32(x) for x in (queries, heads, width, levels, points)],
    )


def reference_impl(value, shapes, locations, weights):
    queries, heads, _, points = weights.shape
    out = np.zeros((queries, heads, value.shape[-1]), dtype=np.float32)
    offset = 0
    for level, (height, width) in enumerate(shapes):
        image = value[offset : offset + height * width].reshape(height, width, heads, -1)
        for q, h, p in np.ndindex(queries, heads, points):
            x, y = locations[q, h, level, p] * [width, height] - 0.5
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            for row, col in ((y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)):
                if 0 <= row < height and 0 <= col < width:
                    weight = (1 - abs(x - col)) * (1 - abs(y - row)) * weights[q, h, level, p]
                    out[q, h] += weight * image[row, col, h]
        offset += height * width
    return out


def dispatch_config(case):
    return case["queries"] * case["heads"] * case["width"], 1, 1, 64, 1, 1, False
