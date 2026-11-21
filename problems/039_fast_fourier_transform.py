"""39. Fast Fourier Transform."""

import numpy as np

RTOL, ATOL = 1e-3, 1e-3
TEST_CASES = [{"n": n} for n in (1, 17, 64, 512)]


def make_inputs(case, rng):
    signal = rng.standard_normal((case["n"], 2), dtype=np.float32)
    return [signal], signal.shape, [np.uint32(case["n"])]


def reference_impl(signal):
    spectrum = np.fft.fft(signal[:, 0] + 1j * signal[:, 1])
    return np.column_stack((spectrum.real, spectrum.imag)).astype(np.float32)


def dispatch_config(case):
    return 64, 1, 1, 64, 1, 1, False
