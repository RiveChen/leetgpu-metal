"""140. FP8 Block-Scaled GEMM."""

import numpy as np

RTOL, ATOL = 0.05, 0.1
TEST_CASES = [
    {"rows": 1, "cols": 1, "blocks": 1},
    {"rows": 5, "cols": 7, "blocks": 2},
    {"rows": 9, "cols": 13, "blocks": 3},
]


def make_inputs(case, rng):
    rows = case["rows"]
    cols = case["cols"]
    blocks = case["blocks"]

    def quantized(size):
        # Include zeros, subnormals and finite exponent-15 encodings of both signs.
        payload = rng.integers(0, 127, (size, blocks * 128), dtype=np.uint8)
        payload |= rng.integers(0, 2, payload.shape, dtype=np.uint8) << 7
        payload[0, :8] = [0, 128, 1, 129, 126, 254, 56, 184]
        scale = rng.uniform(0.015, 0.08, (size, blocks)).astype(np.float32)
        return payload, scale

    x, xscale = quantized(rows)
    w, wscale = quantized(cols)
    return (
        [x, xscale, w, wscale],
        [((rows, cols), np.float16)],
        [np.uint32(v) for v in (rows, cols, 128 * blocks)],
    )


def reference_impl(x, xscale, w, wscale):
    def unpack(bits, scales):
        exponent = ((bits >> 3) & 15).astype(int)
        fraction = (bits & 7).astype(np.float64) / 8
        magnitude = np.where(
            exponent, np.exp2(exponent - 7) * (1 + fraction), np.exp2(-6) * fraction
        )
        signed = np.where(bits & 128, -magnitude, magnitude)
        return signed * np.repeat(scales, 128, axis=1)

    return (unpack(x, xscale) @ unpack(w, wscale).T).astype(np.float16)


def dispatch_config(case):
    return case["rows"] * case["cols"], 1, 1, 64, 1, 1, False
