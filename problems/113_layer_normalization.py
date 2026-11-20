import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4

TEST_CASES = [
    {"N": 2, "C": 4, "eps": 1e-5},
    {"N": 32, "C": 128, "eps": 1e-5},
    {"N": 128, "C": 768, "eps": 1e-5},  # BERT base hidden dim
    {"N": 512, "C": 1024, "eps": 1e-5},  # Large model sequence & hidden dim
]

_current_case = {}


def make_inputs(case: dict, rng: np.random.Generator):
    global _current_case
    _current_case = case

    N, C = case["N"], case["C"]
    eps = case.get("eps", 1e-5)

    input_mat = rng.standard_normal((N, C), dtype=DTYPE)
    weight = rng.uniform(0.5, 1.5, size=C).astype(DTYPE)
    bias = rng.uniform(-0.1, 0.1, size=C).astype(DTYPE)
    out_shape = (N, C)
    scalars = [np.float32(eps), np.uint32(N), np.uint32(C)]
    return [input_mat, weight, bias], out_shape, scalars


def reference_impl(input_mat: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    eps = _current_case.get("eps", 1e-5)
    x = input_mat.astype(np.float64)
    w = weight.astype(np.float64)
    b = bias.astype(np.float64)

    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    norm = (x - mean) / np.sqrt(var + eps)
    return (w * norm + b).astype(DTYPE)


def dispatch_config(case: dict):
    N = case["N"]
    # 1 threadgroup per row, 256 threads per threadgroup
    return (N, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N, C = case["N"], case["C"]
    # read input + read weight + read bias + write output = (N*C + 2*C + N*C) * 4 bytes
    total_bytes = float(2 * N * C + 2 * C) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
