import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-5, 1e-5
TILE = 16

TEST_CASES = [
    {"H": 4, "W": 4},
    {"H": 64, "W": 64},
    {"H": 256, "W": 256},
    {"H": 1024, "W": 1024},
]


def make_inputs(case: dict, rng: np.random.Generator):
    H, W = case["H"], case["W"]
    input_mat = rng.standard_normal((H, W), dtype=DTYPE)
    out_shape = (H // 2, W // 2)
    scalars = [np.uint32(H), np.uint32(W)]
    return [input_mat], out_shape, scalars


def reference_impl(input_mat: np.ndarray) -> np.ndarray:
    H, W = input_mat.shape
    out_H, out_W = H // 2, W // 2
    # standard 2x2 max pooling with stride 2
    reshaped = input_mat.reshape(out_H, 2, out_W, 2)
    return reshaped.max(axis=(1, 3)).astype(DTYPE)


def dispatch_config(case: dict):
    H, W = case["H"], case["W"]
    out_H, out_W = H // 2, W // 2
    grid_x = (out_W + TILE - 1) // TILE
    grid_y = (out_H + TILE - 1) // TILE
    return (grid_x, grid_y, 1, TILE, TILE, 1)


def metrics(case: dict, median_ms: float) -> str:
    H, W = case["H"], case["W"]
    # 4 reads + 1 write per output element = (4 + 1) * (H/2 * W/2) * 4 bytes
    total_bytes = 5.0 * float((H // 2) * (W // 2)) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
