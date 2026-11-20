import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4

TEST_CASES = [
    {"N": 4},
    {"N": 32},
    {"N": 256},
    {"N": 1024},
    {"N": 4096},
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    # use small uniform floats to avoid loss of precision on long accumulative additions
    input_vec = rng.uniform(-1.0, 1.0, size=N).astype(DTYPE)
    out_shape = (N,)
    scalars = [np.uint32(N)]
    return [input_vec], out_shape, scalars


def reference_impl(input_vec: np.ndarray) -> np.ndarray:
    return np.cumsum(input_vec.astype(np.float64)).astype(DTYPE)


def dispatch_config(case: dict):
    # Single threadgroup stride processing for starter inclusive scan
    return (1, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 1 read + 1 write = 2 * N * 4 bytes
    total_bytes = 2.0 * float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
