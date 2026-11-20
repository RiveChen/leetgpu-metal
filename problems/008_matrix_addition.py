import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-5, 1e-5
TILE = 16

TEST_CASES = [
    {"N": 2},
    {"N": 31},  # non-multiple of 16/32
    {"N": 129},  # prime dimension
    {"N": 1024},  # 1K square
    {"N": 4096},  # 4K square
]


def make_inputs(case: dict, rng: np.random.Generator):
    N = case["N"]
    A = rng.standard_normal((N, N), dtype=DTYPE)
    B = rng.standard_normal((N, N), dtype=DTYPE)
    out_shape = (N, N)
    scalars = [np.uint32(N)]
    return [A, B], out_shape, scalars


def reference_impl(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A + B


def dispatch_config(case: dict):
    N = case["N"]
    grid_x = (N + TILE - 1) // TILE
    grid_y = (N + TILE - 1) // TILE
    return (grid_x, grid_y, 1, TILE, TILE, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 2 reads + 1 write = 3 * N * N * 4 bytes
    total_bytes = 3.0 * float(N * N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
