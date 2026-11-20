import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4
TILE = 16

TEST_CASES = [
    {"H": 5, "W": 5, "kH": 3, "kW": 3},
    {"H": 64, "W": 64, "kH": 3, "kW": 3},
    {"H": 128, "W": 128, "kH": 5, "kW": 5},
    {"H": 256, "W": 256, "kH": 7, "kW": 7},
]


def make_inputs(case: dict, rng: np.random.Generator):
    H, W, kH, kW = case["H"], case["W"], case["kH"], case["kW"]
    input_mat = rng.standard_normal((H, W), dtype=DTYPE)
    kernel_mat = rng.standard_normal((kH, kW), dtype=DTYPE)
    out_shape = (H - kH + 1, W - kW + 1)
    scalars = [np.uint32(H), np.uint32(W), np.uint32(kH), np.uint32(kW)]
    return [input_mat, kernel_mat], out_shape, scalars


def reference_impl(input_mat: np.ndarray, kernel_mat: np.ndarray) -> np.ndarray:
    H, W = input_mat.shape
    kH, kW = kernel_mat.shape
    out_H = H - kH + 1
    out_W = W - kW + 1
    out = np.zeros((out_H, out_W), dtype=DTYPE)
    for r in range(out_H):
        for c in range(out_W):
            out[r, c] = np.sum(input_mat[r : r + kH, c : c + kW] * kernel_mat)
    return out


def dispatch_config(case: dict):
    H, W, kH, kW = case["H"], case["W"], case["kH"], case["kW"]
    out_H = H - kH + 1
    out_W = W - kW + 1
    grid_x = (out_W + TILE - 1) // TILE
    grid_y = (out_H + TILE - 1) // TILE
    return (grid_x, grid_y, 1, TILE, TILE, 1)


def metrics(case: dict, median_ms: float) -> str:
    H, W, kH, kW = case["H"], case["W"], case["kH"], case["kW"]
    out_H = H - kH + 1
    out_W = W - kW + 1
    flops = 2.0 * float(out_H * out_W * kH * kW)
    gflops = flops / (median_ms * 1e6)
    return f"{gflops:.2f} GFLOP/s"
