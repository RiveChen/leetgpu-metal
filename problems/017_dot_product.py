import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4

TEST_CASES = [
    {"N": 4},
    {"N": 1023},  # prime dimension
    {"N": 65536},  # 64K
    {"N": 1048576},  # 1M
    {"N": 16777216},  # 16M
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    A = rng.uniform(-1.0, 1.0, size=N).astype(DTYPE)
    B = rng.uniform(-1.0, 1.0, size=N).astype(DTYPE)
    out_shape = (1,)
    scalars = [np.uint32(N)]
    return [A, B], out_shape, scalars


def reference_impl(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.array([np.dot(A.astype(np.float64), B.astype(np.float64))], dtype=DTYPE)


def dispatch_config(case: dict):
    return (1, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 2 reads = 2 * N * 4 bytes
    total_bytes = 2.0 * float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
