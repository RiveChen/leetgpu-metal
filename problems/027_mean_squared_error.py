import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4

TEST_CASES = [
    {"N": 4},
    {"N": 1024},
    {"N": 65536},
    {"N": 1048576},
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    pred = rng.standard_normal(N, dtype=DTYPE)
    target = rng.standard_normal(N, dtype=DTYPE)
    out_shape = (1,)
    scalars = [np.uint32(N)]
    return [pred, target], out_shape, scalars


def reference_impl(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    return np.array([np.mean(diff**2)], dtype=DTYPE)


def dispatch_config(case: dict):
    return (1, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 2 reads = 2 * N * 4 bytes
    total_bytes = 2.0 * float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
