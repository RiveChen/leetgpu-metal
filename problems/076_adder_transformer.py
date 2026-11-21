"""76. Adder Transformer Inference."""

import numpy as np

# Absolute error distinguishes nearby digit logits despite their large shared offset.
RTOL, ATOL = 0.0, 0.04

TEST_CASES = [{"batch": 1}, {"batch": 3}, {"batch": 17}]


def unit_rms(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)


def silu(x):
    return x * np.exp(-np.logaddexp(0, -x))


def make_inputs(case, rng):
    batch = case["batch"]
    prompts = np.zeros((batch, 31), dtype=np.int32)
    for start in (1, 20):
        prompts[:, start : start + 10] = rng.integers(0, 10, (batch, 10), dtype=np.int32)
    # Every batch includes a carry chain.
    prompts[0, 1:11] = 9
    prompts[0, 20:30] = 0
    prompts[0, 20] = 1
    angle = 10.3 * (2 * np.pi / 19)
    weights = np.array(
        [
            1000,
            0.001,
            np.cos(angle),
            -np.sin(angle),
            -22000 / np.sqrt(2),
            -94 * 128,
            128000,
            100 / 256,
            100 / np.sqrt(2),
            -20 / np.sqrt(2),
        ],
        np.float32,
    )
    return [prompts, weights], (batch, 11, 10), [np.uint32(batch)]


def reference_impl(prompts, w):
    digits = np.arange(10, dtype=np.float64)
    table = np.column_stack((w[0] - w[1] * digits**2, -digits))
    omega = 2 * np.pi / 19
    scale = np.log(10) / (2 * (np.cos(0.3 * omega) - np.cos(0.7 * omega)))
    sequence = prompts.copy()
    result = np.empty((len(sequence), 11, 10), np.float32)
    for step in range(11):
        raw = table[sequence]
        embedded = unit_rms(raw)
        query = unit_rms(embedded[:, -1, :1] * w[2:4])
        key_x = unit_rms(np.stack((embedded[..., 0], np.zeros_like(embedded[..., 0])), axis=-1))[
            ..., 0
        ]
        # Relative RoPE rotation of each key as seen by the final query.
        theta = omega * (np.arange(sequence.shape[1]) - (sequence.shape[1] - 1))
        scores = scale * key_x * (query[:, :1] * np.cos(theta) + query[:, 1:] * np.sin(theta))
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        h = raw[:, -1, :].copy()
        h[:, 1] += (weights * (embedded[..., 1] * w[4])).sum(axis=1) / weights.sum(axis=1)
        base, carry = unit_rms(h).T
        gate = base * w[5] + carry * w[6]
        other = gate - base * w[6] / 1000
        h[:, 1] += w[7] * base * (silu(other) - silu(gate))
        logits = (unit_rms(h) * w[8:10]) @ table.T
        result[:, step] = logits
        sequence = np.column_stack((sequence, logits.argmax(axis=1)))
    return result


def dispatch_config(case):
    return case["batch"], 1, 1, 64, 1, 1, False
