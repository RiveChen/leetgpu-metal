import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-5, 1e-5

TEST_CASES = [
    {"N": 4},
    {"N": 31},
    {"N": 100_003},
    {"N": 16_777_216},
    {"N": 67_108_864},
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    input_vec = rng.standard_normal(N, dtype=DTYPE)
    out_shape = (N,)
    scalars = [np.uint32(N)]
    return [input_vec], out_shape, scalars


def reference_impl(input_vec: np.ndarray) -> np.ndarray:
    x = input_vec.astype(np.float64)
    # silu = x / (1 + exp(-x))
    sig = 1.0 / (1.0 + np.exp(-x))
    return (x * sig).astype(DTYPE)


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
