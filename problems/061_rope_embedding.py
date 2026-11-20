import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4
TILE = 16

TEST_CASES = [
    {"M": 4, "D": 8},
    {"M": 128, "D": 64},  # typical attention head
    {"M": 512, "D": 128},  # LLaMA head dim D=128
    {"M": 2048, "D": 128},  # 2K context length
    {"M": 8192, "D": 128},  # 8K context length
]


def make_inputs(case: dict, rng: np.random.Generator):
    M, D = case["M"], case["D"]
    Q = rng.standard_normal((M, D), dtype=DTYPE)
    cos = rng.uniform(-1.0, 1.0, size=(M, D)).astype(DTYPE)
    sin = rng.uniform(-1.0, 1.0, size=(M, D)).astype(DTYPE)
    out_shape = (M, D)
    scalars = [np.uint32(M), np.uint32(D)]
    return [Q, cos, sin], out_shape, scalars


def reference_impl(Q: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    _, D = Q.shape
    half = D // 2
    x1 = Q[:, :half]
    x2 = Q[:, half:]
    rotated_Q = np.concatenate((-x2, x1), axis=-1)
    return (Q * cos + rotated_Q * sin).astype(DTYPE)


def dispatch_config(case: dict):
    M, D = case["M"], case["D"]
    grid_x = (D + TILE - 1) // TILE
    grid_y = (M + TILE - 1) // TILE
    return (grid_x, grid_y, 1, TILE, TILE, 1)


def metrics(case: dict, median_ms: float) -> str:
    M, D = case["M"], case["D"]
    # 3 reads (Q, cos, sin) + 1 write = 4 * M * D * 4 bytes
    total_bytes = 4.0 * float(M * D) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
