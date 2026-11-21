"""28. Gaussian Blur."""

import numpy as np

RTOL, ATOL = 1e-05, 1e-05
TEST_CASES = [
    {"rows": 1, "cols": 1, "kh": 3, "kw": 3},
    {"rows": 7, "cols": 13, "kh": 1, "kw": 1},
    {"rows": 43, "cols": 61, "kh": 3, "kw": 5},
]


def make_inputs(case, rng):
    rows, cols, kh, kw = case["rows"], case["cols"], case["kh"], case["kw"]
    image = rng.standard_normal((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:kh, :kw]
    kernel = np.exp(-((y - kh // 2) ** 2 + (x - kw // 2) ** 2) / 2)
    kernel = (kernel / kernel.sum()).astype(np.float32)
    return [image, kernel], image.shape, [np.uint32(v) for v in (*image.shape, *kernel.shape)]


def reference_impl(image, kernel):
    rh, rw = (size // 2 for size in kernel.shape)
    padded = np.pad(image, ((rh, rh), (rw, rw)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return np.einsum("ijab,ab->ij", windows, kernel)


def dispatch_config(case):
    return case["rows"] * case["cols"], 1, 1, 64, 1, 1, False
