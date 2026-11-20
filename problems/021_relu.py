import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-5, 1e-5

TEST_CASES = [
    {"N": 1},  # single element edge case
    {"N": 31},  # non SIMD-group divisible
    {"N": 100_003},  # prime elements
    {"N": 16_777_216},  # 16M elements (~128 MB total I/O)
    {"N": 67_108_864},  # 64M elements (~512 MB total I/O, tests peak memory bandwidth)
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    input_vec = rng.standard_normal(N, dtype=DTYPE)
    out_shape = (N,)
    scalars = [np.uint32(N)]
    return [input_vec], out_shape, scalars


def reference_impl(input_vec: np.ndarray) -> np.ndarray:
    return np.maximum(input_vec, 0.0).astype(DTYPE)


def dispatch_config(case: dict):
    N = case["N"]
    TG = 256
    return ((N + TG - 1) // TG, 1, 1, TG, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 1 read + 1 write = 2 * N * 4 bytes
    total_bytes = 2.0 * float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
