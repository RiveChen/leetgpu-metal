"""127. Fused AdamW. Updated optimizer state is written to separate output buffers."""

import numpy as np

RTOL, ATOL = 0.0001, 0.0001
TEST_CASES = [
    {"n": 1, "step": 1, "lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-08, "decay": 0.01},
    {"n": 37, "step": 5, "lr": 0.02, "beta1": 0.8, "beta2": 0.97, "eps": 1e-06, "decay": 0.0},
    {
        "n": 1027,
        "step": 83,
        "lr": 0.003,
        "beta1": 0.94,
        "beta2": 0.987,
        "eps": 1e-08,
        "decay": 0.05,
    },
]

REFERENCE_USES_SCALARS = True


def make_inputs(case, rng):
    n = case["n"]
    step = case["step"]
    lr = case["lr"]
    beta1 = case["beta1"]
    beta2 = case["beta2"]
    eps = case["eps"]
    decay = case["decay"]
    params = rng.standard_normal(n, dtype=np.float32)
    grad = rng.standard_normal(n, dtype=np.float32)
    m = rng.standard_normal(n, dtype=np.float32) if step > 1 else np.zeros_like(grad)
    v = rng.random(n, dtype=np.float32) if step > 1 else np.zeros_like(grad)
    return (
        [params, grad, m, v],
        [(params.shape, np.float32)] * 3,
        [
            np.uint32(n),
            *[np.float32(value) for value in (lr, beta1, beta2, eps, decay)],
            np.uint32(step),
        ],
    )


def reference_impl(params, grad, m, v, n, lr, beta1, beta2, eps, decay, step):
    next_m = beta1 * m + (1 - beta1) * grad
    next_v = beta2 * v + (1 - beta2) * grad**2
    mhat = next_m / (1 - float(beta1) ** int(step))
    vhat = next_v / (1 - float(beta2) ** int(step))
    return params * (1 - lr * decay) - lr * mhat / (np.sqrt(vhat) + eps), next_m, next_v


def dispatch_config(case):
    return case["n"], 1, 1, 64, 1, 1, False
