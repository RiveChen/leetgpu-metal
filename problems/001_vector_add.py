import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-5, 1e-5

TEST_CASES = [
    {"N": 31},  # non SIMD-group(32) divisible boundary test
    {"N": 100_003},  # prime dimension overflow test
    {"N": 16_777_216},  # 16M elements (~192 MB total memory access, test memory bandwidth limit)
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    A = rng.standard_normal(N, dtype=DTYPE)
    B = rng.standard_normal(N, dtype=DTYPE)
    out_shape = (N,)
    scalars = [np.uint32(N)]
    return [A, B], out_shape, scalars


def reference_impl(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A + B


def dispatch_config(case: dict):
    N = case["N"]
    TG = 256
    return ((N + TG - 1) // TG, 1, 1, TG, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # read A + read B + write C = 3 * N * 4 bytes
    total_bytes = 3.0 * N * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
