"""78. 2D FFT."""

import numpy as np

RTOL, ATOL = 0.01, 0.01

TEST_CASES = [{"h": 1, "w": 5}, {"h": 7, "w": 8}, {"h": 16, "w": 32}]


def make_inputs(case, rng):
    h, w = case["h"], case["w"]
    x = rng.standard_normal((h, w, 2), dtype=np.float32)
    return (
        [x],
        x.shape,
        [np.uint32(v) for v in (h, w)],
    )


def reference_impl(x):
    spectrum = np.fft.fft2(x[..., 0] + 1j * x[..., 1])
    return np.stack((spectrum.real, spectrum.imag), axis=-1).astype(np.float32)


def dispatch_config(case):
    return case["w"], 1, 1, 64, 1, 1, False


def workspace_bytes(case):
    return 8 * case["h"] * case["w"]
