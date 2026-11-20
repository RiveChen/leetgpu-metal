import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-3, 1e-3
TILE = 16

TEST_CASES = [
    {
        "M": 4,
        "N": 4,
        "K": 4,
    },  # tiny dims (less than single Tile, test local out-of-bounds within group)
    {
        "M": 129,
        "N": 71,
        "K": 53,
    },  # prime dims (test Tiled boundary padding 0 and Barrier safety)
    {
        "M": 512,
        "N": 512,
        "K": 512,
    },  # medium regular dimensions
    {
        "M": 1024,
        "N": 1024,
        "K": 1024,
    },  # 1K square (~2.15 GFLOPs)
    {
        "M": 2048,
        "N": 2048,
        "K": 2048,
    },  # 2K square (~17.18 GFLOPs, test compute performance)
    {
        "M": 4096,
        "N": 4096,
        "K": 4096,
    },  # 4K square (~137.44 GFLOPs, test memory bandwidth limit)
]


def make_inputs(case: dict, rng: np.random.Generator):
    M, N, K = case["M"], case["N"], case["K"]
    A = rng.standard_normal((M, K), dtype=DTYPE)
    B = rng.standard_normal((K, N), dtype=DTYPE)
    out_shape = (M, N)
    scalars = [np.uint32(M), np.uint32(N), np.uint32(K)]
    return [A, B], out_shape, scalars


def reference_impl(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A.astype(np.float64) @ B.astype(np.float64)).astype(DTYPE)


def dispatch_config(case: dict):
    M, N = case["M"], case["N"]
    grid_x = (N + TILE - 1) // TILE
    grid_y = (M + TILE - 1) // TILE
    return (grid_x, grid_y, 1, TILE, TILE, 1)


def metrics(case: dict, median_ms: float) -> str:
    M, N, K = case["M"], case["N"], case["K"]
    flops = 2.0 * M * N * K
    gflops = flops / (median_ms * 1e6)
    return f"{gflops:.2f} GFLOP/s"
