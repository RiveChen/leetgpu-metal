"""66. RGB to Grayscale."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05

TEST_CASES = [{"width": w, "height": h} for w, h in ((1, 1), (23, 7), (513, 317))]


def make_inputs(case, rng):
    w, h = case["width"], case["height"]
    x = rng.uniform(0, 255, (h, w, 3)).astype(np.float32)
    return [x], (h, w), [np.uint32(w), np.uint32(h)]


def reference_impl(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def dispatch_config(case):
    return case["width"] * case["height"], 1, 1, 64, 1, 1, False
