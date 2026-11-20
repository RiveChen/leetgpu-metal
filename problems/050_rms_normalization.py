import numpy as np

ENTRY = "solve"
DTYPE = np.float32
RTOL, ATOL = 1e-4, 1e-4

TEST_CASES = [
    {"N": 4, "gamma": 1.0, "beta": 0.0, "eps": 1e-5},
    {"N": 768, "gamma": 1.2, "beta": 0.1, "eps": 1e-5},  # small transformer hidden dim
    {"N": 1024, "gamma": 0.9, "beta": -0.05, "eps": 1e-5},
    {"N": 2048, "gamma": 1.0, "beta": 0.0, "eps": 1e-6},
    {"N": 4096, "gamma": 1.1, "beta": 0.0, "eps": 1e-6},  # LLaMA 7B hidden dim
    {"N": 8192, "gamma": 1.0, "beta": 0.0, "eps": 1e-6},  # large model hidden dim
]

_current_case = {}


def make_inputs(case: dict, rng: np.random.Generator):
    global _current_case
    _current_case = case

    N = case["N"]
    gamma = case.get("gamma", 1.0)
    beta = case.get("beta", 0.0)
    eps = case.get("eps", 1e-5)

    input_vec = rng.standard_normal(N, dtype=DTYPE)
    out_shape = (N,)
    scalars = [np.float32(gamma), np.float32(beta), np.float32(eps), np.uint32(N)]
    return [input_vec], out_shape, scalars


def reference_impl(input_vec: np.ndarray) -> np.ndarray:
    gamma = _current_case.get("gamma", 1.0)
    beta = _current_case.get("beta", 0.0)
    eps = _current_case.get("eps", 1e-5)

    x = input_vec.astype(np.float64)
    rms = np.sqrt(np.mean(x**2) + eps)
    normalized = x / rms
    return (gamma * normalized + beta).astype(DTYPE)


def dispatch_config(case: dict):
    # Launch 1 threadgroup of 256 threads to perform reduction & normalization
    return (1, 1, 1, 256, 1, 1)


def metrics(case: dict, median_ms: float) -> str:
    N = case["N"]
    # 1 read (sum of squares) + 1 read/write (normalize & write) = ~3 * N * 4 bytes
    total_bytes = 3.0 * float(N) * 4.0
    gb_s = total_bytes / (median_ms * 1e6)
    return f"{gb_s:.2f} GB/s"
