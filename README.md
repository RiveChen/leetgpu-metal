# LeetGPU in Metal

English · [简体中文](README.zh-CN.md)

Write Metal compute kernels on Apple Silicon Macs, verify numerical correctness with NumPy, and benchmark raw GPU execution time. Each problem includes a Python specification, a starter kernel, and an AI-generated reference solution. The C++ runtime compiles shaders just-in-time, allowing you to iterate and re-benchmark immediately with a single command.

---

## Quick Start

### Prerequisites

- An Apple Silicon Mac (M-series) running macOS 13.0 or later (Metal MathMode requires macOS 15+).
- Xcode Command Line Tools (`xcode-select --install`) with C++17 support.
- CMake 3.20+ and Ninja (`brew install cmake ninja`).
- Python 3.12+ with NumPy (`pip install 'numpy>=2.5.5'`).

### Build & Run

Clone the repository with submodules (includes Apple's `metal-cpp` headers):

```sh
git clone --recurse-submodules https://github.com/RiveChen/leetgpu-metal.git
cd leetgpu-metal

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install 'numpy>=2.5.5'

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build

python3 scripts/bench.py 001_vector_add --shader shaders/solutions/001_vector_add.metal
```

> **Note**: If already cloned without submodules, run `git submodule update --init --recursive` before building.

The command above checks the installation using the reference solution. To practice, implement the TODO in `shaders/starters/001_vector_add.metal` and run the default command below. The untouched starter compiles and fails output validation.

```sh
python3 scripts/bench.py 001_vector_add
```

### Run with uv

If you use Homebrew, install uv with `brew install uv`. Skip the venv and pip steps above and build the native library with CMake as shown. uv reads the Python version and NumPy requirement from `scripts/bench.py`. Run the reference solution first, then try your completed starter.

```sh
uv run scripts/bench.py 001_vector_add --shader shaders/solutions/001_vector_add.metal
uv run scripts/bench.py 001_vector_add
```

---

## Built-in Problems

| Problem | Description | Metric |
| --- | --- | --- |
| [001_vector_add](problems/001_vector_add.py) · [Starter](shaders/starters/001_vector_add.metal) · [Solution](shaders/solutions/001_vector_add.metal) | FP32 element-wise addition across sizes 31, 100,003, and 16,777,216 | Effective bandwidth in GB/s |
| [002_matrix_mul](problems/002_matrix_mul.py) · [Starter](shaders/starters/002_matrix_mul.metal) · [Solution](shaders/solutions/002_matrix_mul.metal) | FP32 general matrix multiplication up to 4096 × 4096 square matrices | Throughput in GFLOP/s (`2 × M × N × K`) |
| [003_matrix_transpose](problems/003_matrix_transpose.py) · [Starter](shaders/starters/003_matrix_transpose.metal) · [Solution](shaders/solutions/003_matrix_transpose.metal) | 2D matrix transpose testing memory coalescing and threadgroup tiling | Effective bandwidth in GB/s |
| [004_reduction](problems/004_reduction.py) · [Starter](shaders/starters/004_reduction.metal) · [Solution](shaders/solutions/004_reduction.metal) | Parallel array summation using SIMD shuffle and threadgroup reduction | Effective bandwidth in GB/s |
| [005_softmax](problems/005_softmax.py) · [Starter](shaders/starters/005_softmax.metal) · [Solution](shaders/solutions/005_softmax.metal) | Numerically stable online / multi-pass Softmax with large-offset tests | Effective bandwidth in GB/s |
| [008_matrix_addition](problems/008_matrix_addition.py) · [Starter](shaders/starters/008_matrix_addition.metal) · [Solution](shaders/solutions/008_matrix_addition.metal) | 2D element-wise matrix addition with 2D grid dispatch | Effective bandwidth in GB/s |
| [010_2d_convolution](problems/010_2d_convolution.py) · [Starter](shaders/starters/010_2d_convolution.metal) · [Solution](shaders/solutions/010_2d_convolution.metal) | 2D valid cross-correlation / convolution with window sliding | Throughput in GFLOP/s |
| [013_histogramming](problems/013_histogramming.py) · [Starter](shaders/starters/013_histogramming.metal) · [Solution](shaders/solutions/013_histogramming.metal) | Histogram binning with hardware atomic additions | Effective bandwidth in GB/s |
| [016_prefix_sum](problems/016_prefix_sum.py) · [Starter](shaders/starters/016_prefix_sum.metal) · [Solution](shaders/solutions/016_prefix_sum.metal) | Inclusive scan prefix sum testing intra-warp and inter-threadgroup scan | Effective bandwidth in GB/s |
| [017_dot_product](problems/017_dot_product.py) · [Starter](shaders/starters/017_dot_product.metal) · [Solution](shaders/solutions/017_dot_product.metal) | Vector inner product using FMA instructions and collective reduction | Effective bandwidth in GB/s |
| [021_relu](problems/021_relu.py) · [Starter](shaders/starters/021_relu.metal) · [Solution](shaders/solutions/021_relu.metal) | Element-wise ReLU activation testing peak memory bandwidth throughput | Effective bandwidth in GB/s |
| [027_mean_squared_error](problems/027_mean_squared_error.py) · [Starter](shaders/starters/027_mean_squared_error.metal) · [Solution](shaders/solutions/027_mean_squared_error.metal) | Mean squared error (MSE) loss computation between prediction and target | Effective bandwidth in GB/s |
| [042_2d_max_pooling](problems/042_2d_max_pooling.py) · [Starter](shaders/starters/042_2d_max_pooling.metal) · [Solution](shaders/solutions/042_2d_max_pooling.metal) | 2D 2x2 max pooling with stride 2 for spatial downsampling | Effective bandwidth in GB/s |
| [050_rms_normalization](problems/050_rms_normalization.py) · [Starter](shaders/starters/050_rms_normalization.metal) · [Solution](shaders/solutions/050_rms_normalization.metal) | Root Mean Square Normalization layer widely used in modern LLMs | Effective bandwidth in GB/s |
| [052_silu](problems/052_silu.py) · [Starter](shaders/starters/052_silu.metal) · [Solution](shaders/solutions/052_silu.metal) | SiLU (Swish) activation function ($x \cdot \sigma(x)$) for modern LLMs | Effective bandwidth in GB/s |
| [061_rope_embedding](problems/061_rope_embedding.py) · [Starter](shaders/starters/061_rope_embedding.metal) · [Solution](shaders/solutions/061_rope_embedding.metal) | Rotary Positional Embedding (RoPE) used in LLaMA, Mistral, and Qwen | Effective bandwidth in GB/s |
| [113_layer_normalization](problems/113_layer_normalization.py) · [Starter](shaders/starters/113_layer_normalization.metal) · [Solution](shaders/solutions/113_layer_normalization.metal) | Layer Normalization (mean, variance, affine scaling) for Transformers | Effective bandwidth in GB/s |

```sh
python3 scripts/bench.py 002_matrix_mul --shader shaders/solutions/002_matrix_mul.metal
```

---

## Benchmark Settings & Mechanics

Each test case runs `warmup` iterations (default: 2) followed by `runs` timed iterations (default: 15). The runner measures kernel execution time directly on the GPU command buffer using hardware timestamps (`GPUStartTime` / `GPUEndTime`), filtering out CPU dispatch and host queueing noise.

Shader JIT compilation, NumPy reference computations, and an optional 64 MiB buffer overwrite (to flush the hardware L2/SLC cache) occur outside the timed execution block.

| Flag | Description |
| --- | --- |
| `--warmup N` | Warmup iterations before timing (default: 2, min: 0). |
| `--runs N` | Timed iterations (default: 15, min: 1). Reports median and minimum ms. |
| `--seed N` | RNG seed for generating inputs (default: 42). |
| `--safe-math` | Compile MSL with safe math mode (default uses fast-math). |
| `--no-flush` | Skip the 64 MiB buffer overwrite designed to mitigate cache reuse. |
| `--entry NAME` | Override the kernel entry function name (default: `ENTRY` or `solve`). |
| `--lib PATH` | Path to the harness dynamic library (default: `build/libleetgpu_metal.dylib`). |

---

## Key Architecture Highlights

- **Zero-Copy Unified Memory (UMA)**: Uses `MTLResourceStorageModeShared` with page-aligned memory mapped via `mmap`, avoiding redundant host-device copies.
- **Canary Padding Defense**: Output buffers are page-aligned with `0xCC` canary bytes appended past the payload. Any GPU out-of-bounds write is immediately flagged.
- **Input Immutability Guard**: Takes an immutable snapshot of input buffers before kernel launch and verifies post-execution that the GPU did not mutate read-only inputs.
- **Uninitialized Output Detection**: Output buffers are pre-filled with `NaN` (floating point) or bitwise-inverted values (integers/booleans) to prevent unwritten elements from falsely passing.

---

## Adding New Problems

Create `problems/<name>.py` and matching `<name>.metal` files in `shaders/starters/` and `shaders/solutions/`, then run with `python3 scripts/bench.py <name>`. See [001_vector_add.py](problems/001_vector_add.py) and [Starter](shaders/starters/001_vector_add.metal) · [Solution](shaders/solutions/001_vector_add.metal) as references.

1. **Test Cases & Inputs**: Define `TEST_CASES` and implement `make_inputs(case, rng)` returning `(inputs, out_spec, scalars)`. For single output, pass `(shape,)` with `DTYPE` (default `np.float32`). For multiple outputs, pass `[(shape, dtype), ...]`.
2. **Reference Implementation**: Implement `reference_impl(*inputs)` using readonly input arrays. Floating-point results must be finite and within `RTOL` / `ATOL` tolerance (defaults: `1e-3`). Integer and boolean results require exact match.
3. **Dispatch Grid**: Implement `dispatch_config(case)` returning `(grid_x, grid_y, grid_z, tg_x, tg_y, tg_z[, use_threadgroups])`. Defaults to threadgroup dispatch; pass `False` as the 7th element if the grid specifies total threads.
4. **Buffer Bindings**: Bind sequentially starting from `[[buffer(0)]]`:
   - `[[buffer(0)]]` ... `[[buffer(k-1)]]`: Input arrays
   - `[[buffer(k)]]` ... `[[buffer(k+m-1)]]`: Output arrays
   - `[[buffer(k+m)]]` ...: Scalar constants / structs
5. **Metrics (Optional)**: Implement `metrics(case, median_ms) -> str` to display problem-specific performance figures (e.g. GB/s, GFLOP/s).

---

## Running Tests

Run the test suite on any Apple Silicon Mac with Metal access:

```sh
python3 -m unittest discover -s tests -v
```

To run the same tests with uv:

```sh
uv run --python ">=3.12" --with "numpy>=2.5.5" python -m unittest discover -s tests -v
```

The tests verify buffer canary detection, mutation defense, invalid CLI arguments, and error reporting.

---

## Code Quality

```sh
# Lint and format Python
ruff check .
ruff format --check .

# Format C++ and Metal shaders
clang-format --dry-run --Werror src/*.cpp src/*.hpp shaders/starters/*.metal shaders/solutions/*.metal
```

---

## Acknowledgments

Thanks to [LeetGPU](https://leetgpu.com/) for inspiring this project’s approach to learning GPU programming through kernel exercises.

## License

This project and Apple's `metal-cpp` submodule are licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [metal-cpp LICENSE](third_party/metal-cpp/LICENSE.txt).
