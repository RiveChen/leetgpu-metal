import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4

TEST_CASES = [
    {"N": 8},
    {"N": 1023},  # non SIMD-group divisible
    {"N": 100_003},  # prime elements
    {"N": 1_048_576},  # 1M elements
    {"N": 16_777_216},  # 16M elements (~64 MB, tests memory bandwidth)
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    # use smaller variance uniform numbers to prevent float32 summation precision drift
    input_vec = rng.uniform(-1.0, 1.0, size=N).astype(DTYPE)
    out_shape = (1,)
    scalars = [np.uint32(N)]
    return [input_vec], out_shape, scalars


def reference_impl(input_vec: np.ndarray) -> np.ndarray:
    return np.array([np.sum(input_vec.astype(np.float64))], dtype=DTYPE)


def dispatch_config(case: dict):
    # Launch 1 threadgroup of 256 threads to perform grid-stride reduction
    return (1, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 1 read = N * 4 bytes
    total_bytes = float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
