"""117. Non-Maximum Suppression."""

import numpy as np

RTOL, ATOL = 0.0, 0.0
TEST_CASES = [
    {"count": 1, "threshold": 0.0},
    {"count": 4, "threshold": 0.5},
    {"count": 135, "threshold": 0.25},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    count = case["count"]
    threshold = case["threshold"]
    origins = rng.integers(-5, 6, (count, 2))
    boxes = np.concatenate([origins, origins + rng.integers(1, 9, (count, 2))], axis=1).astype(
        np.float32
    )
    scores = rng.integers(0, 5, count).astype(np.float32) / 4
    if count >= 4:
        # The first pair ties; the third box has IoU exactly 0.5 with either.
        boxes[:4] = [[0, 0, 4, 2], [0, 0, 4, 2], [0, 0, 2, 2], [8, 0, 9, 1]]
        scores[:4] = 1
    return [boxes, scores], [((count,), np.int32)], [np.uint32(count), np.float32(threshold)]


def reference_impl(boxes, scores, count, threshold):
    pending = np.argsort(-scores, kind="stable")
    keep = np.zeros(count, dtype=np.int32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while pending.size:
        winner = pending[0]
        keep[winner] = 1
        rest = pending[1:]
        wh = np.maximum(
            0,
            np.minimum(boxes[winner, 2:], boxes[rest, 2:])
            - np.maximum(boxes[winner, :2], boxes[rest, :2]),
        )
        intersection = wh.prod(axis=1)
        iou = intersection / (areas[winner] + areas[rest] - intersection)
        pending = rest[iou <= threshold]
    return keep


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
