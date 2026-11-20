import numpy as np

ENTRY = "solve"
DTYPE = np.int32

TEST_CASES = [
    {"N": 16, "num_bins": 4},
    {"N": 1000, "num_bins": 16},
    {"N": 65536, "num_bins": 256},
    {"N": 1048576, "num_bins": 256},
]


def make_inputs(case: dict, rng: np.random.Generator):
    N, num_bins = case["N"], case["num_bins"]
    input_vec = rng.integers(0, num_bins, size=N, dtype=DTYPE)
    out_shape = (num_bins,)
    scalars = [np.uint32(N), np.uint32(num_bins)]
    return [input_vec], out_shape, scalars


def reference_impl(input_vec: np.ndarray) -> np.ndarray:
    num_bins = int(np.max(input_vec)) + 1 if len(input_vec) > 0 else 0
    # count bins
    counts = np.bincount(input_vec, minlength=max(num_bins, 1)).astype(DTYPE)
    return counts


def dispatch_config(case: dict):
    # Single threadgroup stride processing ensures deterministic initialization and reduction
    return (1, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 1 read = N * 4 bytes
    total_bytes = float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
