"""150. Distillation KL Divergence."""

import numpy as np

RTOL, ATOL = 0.001, 0.001
TEST_CASES = [
    {"rows": 1, "vocab": 1, "temperature": 0.5},
    {"rows": 13, "vocab": 5, "temperature": 2.0},
    {"rows": 37, "vocab": 153, "temperature": 0.9},
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    rows = case["rows"]
    vocab = case["vocab"]
    temperature = case["temperature"]
    teacher = rng.uniform(-40, 40, (rows, vocab)).astype(np.float32)
    student = rng.uniform(-40, 40, teacher.shape).astype(np.float32)
    student[0] = teacher[0]
    return [teacher, student], (rows,), [np.uint32(rows), np.uint32(vocab), np.float32(temperature)]


def reference_impl(teacher, student, rows, vocab, temperature):
    def log_probability(x):
        z = (x.astype(np.float64) - x.max(axis=1, keepdims=True)) / temperature
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    lp = log_probability(teacher)
    return np.sum(np.exp(lp) * (lp - log_probability(student)), axis=1).astype(np.float32)


def dispatch_config(case):
    return case["rows"], 1, 1, 64, 1, 1, False
