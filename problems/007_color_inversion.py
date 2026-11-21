"""7. Color Inversion."""

import numpy as np

IN_PLACE = True
DTYPE = np.uint8

TEST_CASES = [{"width": w, "height": h} for w, h in ((1, 1), (13, 7), (257, 191))]


def make_inputs(case, rng):
    w, h = case["width"], case["height"]
    image = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    return [image], image.shape, [np.uint32(w), np.uint32(h)]


def reference_impl(image):
    result = image.copy()
    result[..., :3] = 255 - image[..., :3]
    return result


def dispatch_config(case):
    return case["width"] * case["height"], 1, 1, 64, 1, 1, False
