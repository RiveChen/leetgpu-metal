import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4

TEST_CASES = [
    {"N": 3},
    {"N": 128, "offset": 500.0},  # test large offset numerical stability (prevent exp overflow)
    {"N": 1024},
    {"N": 4096},
    {"N": 16384},
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    offset = case.get("offset", 0.0)
    input_vec = (rng.standard_normal(N, dtype=DTYPE) + offset).astype(DTYPE)
    out_shape = (N,)
    scalars = [np.uint32(N)]
    return [input_vec], out_shape, scalars


def reference_impl(input_vec: np.ndarray) -> np.ndarray:
    x = input_vec.astype(np.float64)
    shift_x = x - np.max(x)
    exps = np.exp(shift_x)
    return (exps / np.sum(exps)).astype(DTYPE)


def dispatch_config(case: dict):
    # Launch 1 threadgroup of 256 threads to execute multi-pass softmax
    return (1, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 2 reads (find max, find sum) + 1 read/write (compute exp/sum) = ~4 * N * 4 bytes
    total_bytes = 4.0 * float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
