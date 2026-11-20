import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-5, 1e-5
TILE = 16

TEST_CASES = [
    {"rows": 2, "cols": 3},
    {"rows": 32, "cols": 1024},
    {"rows": 1024, "cols": 32},
    {"rows": 137, "cols": 251},  # prime non-aligned dimensions
    {"rows": 1024, "cols": 1024},  # 1K square matrix
    {"rows": 2048, "cols": 2048},  # 2K square matrix
]


def make_inputs(case: dict, rng: np.random.Generator):
    rows, cols = case["rows"], case["cols"]
    input_mat = rng.standard_normal((rows, cols), dtype=DTYPE)
    out_shape = (cols, rows)
    scalars = [np.uint32(rows), np.uint32(cols)]
    return [input_mat], out_shape, scalars


def reference_impl(input_mat: np.ndarray) -> np.ndarray:
    return input_mat.T.copy()


def dispatch_config(case: dict):
    rows, cols = case["rows"], case["cols"]
    grid_x = (cols + TILE - 1) // TILE
    grid_y = (rows + TILE - 1) // TILE
    return (grid_x, grid_y, 1, TILE, TILE, 1)


def metrics(case: dict, median_ms: float) -> str:
    rows, cols = case["rows"], case["cols"]
    # 1 read + 1 write = 2 * rows * cols * 4 bytes
    total_bytes = 2.0 * rows * cols * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
