# LeetGPU in Metal

English · [简体中文](README.zh-CN.md)

Write Metal compute kernels on Apple Silicon Macs, verify numerical correctness with NumPy, and benchmark raw GPU execution time. Each problem includes a Python specification, a starter kernel, and an AI-generated reference solution. The C++ runtime compiles shaders just-in-time, allowing you to iterate and re-benchmark immediately with a single command.

## TL;DR

Start on an Apple Silicon Mac with Xcode Command Line Tools, CMake, and uv installed.

```sh
git clone --recurse-submodules https://github.com/RiveChen/leetgpu-metal.git
cd leetgpu-metal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Open shaders/starters/001_vector_add.metal and implement the TODO in solve.
uv run scripts/bench.py 001_vector_add
```

---

## Quick Start

### Prerequisites

- An Apple Silicon Mac (M-series) running macOS 13.0 or later (Metal MathMode requires macOS 15+).
- Xcode Command Line Tools (`xcode-select --install`) with C++17 support.
- CMake 3.20+ (`brew install cmake`). Ninja is optional.
- Python 3.12+ with NumPy (`pip install 'numpy>=2.5.5'`).

### Build & Run

Clone the repository with submodules (includes Apple's `metal-cpp` headers):

```sh
git clone --recurse-submodules https://github.com/RiveChen/leetgpu-metal.git
cd leetgpu-metal

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install 'numpy>=2.5.5'

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
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

### Easy

Numbers and titles follow the [upstream Easy catalog](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy). The Metal implementations and test inputs are local to this project.

| Upstream exercise | Problem | Description | Metric |
| --- | --- | --- | --- |
| [1. Vector Addition](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/1_vector_add) | [001_vector_add](problems/001_vector_add.py) · [Starter](shaders/starters/001_vector_add.metal) · [Solution](shaders/solutions/001_vector_add.metal) | FP32 element-wise addition across sizes 31, 100,003, and 16,777,216 | Effective bandwidth in GB/s |
| [2. Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/2_matrix_multiplication) | [002_matrix_multiplication](problems/002_matrix_multiplication.py) · [Starter](shaders/starters/002_matrix_multiplication.metal) · [Solution](shaders/solutions/002_matrix_multiplication.metal) | FP32 general matrix multiplication up to 4096 × 4096 square matrices | Throughput in GFLOP/s (`2 × M × N × K`) |
| [3. Matrix Transpose](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/3_matrix_transpose) | [003_matrix_transpose](problems/003_matrix_transpose.py) · [Starter](shaders/starters/003_matrix_transpose.metal) · [Solution](shaders/solutions/003_matrix_transpose.metal) | 2D matrix transpose testing memory coalescing and threadgroup tiling | Effective bandwidth in GB/s |
| [7. Color Inversion](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/7_color_inversion) | [007_color_inversion](problems/007_color_inversion.py) · [Starter](shaders/starters/007_color_inversion.metal) · [Solution](shaders/solutions/007_color_inversion.metal) | Invert RGB in place and retain alpha | GPU ms |
| [8. Matrix Addition](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/8_matrix_addition) | [008_matrix_addition](problems/008_matrix_addition.py) · [Starter](shaders/starters/008_matrix_addition.metal) · [Solution](shaders/solutions/008_matrix_addition.metal) | 2D element-wise matrix addition with 2D grid dispatch | Effective bandwidth in GB/s |
| [9. 1D Convolution](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/9_1d_convolution) | [009_1d_convolution](problems/009_1d_convolution.py) · [Starter](shaders/starters/009_1d_convolution.metal) · [Solution](shaders/solutions/009_1d_convolution.metal) | Valid 1D cross-correlation | GPU ms |
| [19. Reverse Array](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/19_reverse_array) | [019_reverse_array](problems/019_reverse_array.py) · [Starter](shaders/starters/019_reverse_array.metal) · [Solution](shaders/solutions/019_reverse_array.metal) | Reverse an FP32 array in place | GPU ms |
| [21. ReLU](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/21_relu) | [021_relu](problems/021_relu.py) · [Starter](shaders/starters/021_relu.metal) · [Solution](shaders/solutions/021_relu.metal) | Element-wise ReLU activation testing peak memory bandwidth throughput | Effective bandwidth in GB/s |
| [23. Leaky ReLU](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/23_leaky_relu) | [023_leaky_relu](problems/023_leaky_relu.py) · [Starter](shaders/starters/023_leaky_relu.metal) · [Solution](shaders/solutions/023_leaky_relu.metal) | Apply a negative slope of 0.01 | GPU ms |
| [24. Rainbow Table](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/24_rainbow_table) | [024_rainbow_table](problems/024_rainbow_table.py) · [Starter](shaders/starters/024_rainbow_table.metal) · [Solution](shaders/solutions/024_rainbow_table.metal) | Iterate 32-bit FNV-1a on four-byte words | GPU ms |
| [31. Matrix Copy](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/31_matrix_copy) | [031_matrix_copy](problems/031_matrix_copy.py) · [Starter](shaders/starters/031_matrix_copy.metal) · [Solution](shaders/solutions/031_matrix_copy.metal) | Copy a square FP32 matrix | GPU ms |
| [41. Simple Inference](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/41_simple_inference) | [041_simple_inference](problems/041_simple_inference.py) · [Starter](shaders/starters/041_simple_inference.metal) · [Solution](shaders/solutions/041_simple_inference.metal) | Batched linear inference with explicit weights and bias (zero bias for a bias-free layer) | GPU ms |
| [52. Sigmoid Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/52_silu) | [052_silu](problems/052_silu.py) · [Starter](shaders/starters/052_silu.metal) · [Solution](shaders/solutions/052_silu.metal) | SiLU (Swish) activation function ($x \cdot \sigma(x)$) for modern LLMs | Effective bandwidth in GB/s |
| [54. Swish-Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/54_swiglu) | [054_swiglu](problems/054_swiglu.py) · [Starter](shaders/starters/054_swiglu.metal) · [Solution](shaders/solutions/054_swiglu.metal) | Multiply SiLU of the first half by the second half | GPU ms |
| [62. Value Clipping](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/62_value_clipping) | [062_value_clipping](problems/062_value_clipping.py) · [Starter](shaders/starters/062_value_clipping.metal) · [Solution](shaders/solutions/062_value_clipping.metal) | Clamp to per-case scalar bounds | GPU ms |
| [63. Interleave Arrays](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/63_interleave) | [063_interleave](problems/063_interleave.py) · [Starter](shaders/starters/063_interleave.metal) · [Solution](shaders/solutions/063_interleave.metal) | Alternate elements from two FP32 arrays | GPU ms |
| [65. Gaussian Error Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/65_geglu) | [065_geglu](problems/065_geglu.py) · [Starter](shaders/starters/065_geglu.metal) · [Solution](shaders/solutions/065_geglu.metal) | Gate the first half with erf-based GELU of the second half | GPU ms |
| [66. RGB to Grayscale](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/66_rgb_to_grayscale) | [066_rgb_to_grayscale](problems/066_rgb_to_grayscale.py) · [Starter](shaders/starters/066_rgb_to_grayscale.metal) · [Solution](shaders/solutions/066_rgb_to_grayscale.metal) | Convert packed RGB pixels to grayscale | GPU ms |
| [68. Sigmoid Activation](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/68_sigmoid) | [068_sigmoid](problems/068_sigmoid.py) · [Starter](shaders/starters/068_sigmoid.metal) · [Solution](shaders/solutions/068_sigmoid.metal) | Evaluate the logistic sigmoid | GPU ms |
| [124. GELU Activation](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/124_gelu) | [124_gelu](problems/124_gelu.py) · [Starter](shaders/starters/124_gelu.metal) · [Solution](shaders/solutions/124_gelu.metal) | Evaluate GELU using the Gaussian CDF | GPU ms |

### Other exercises

| Problem | Description | Metric |
| --- | --- | --- |
| [004_reduction](problems/004_reduction.py) · [Starter](shaders/starters/004_reduction.metal) · [Solution](shaders/solutions/004_reduction.metal) | Parallel array summation using SIMD shuffle and threadgroup reduction | Effective bandwidth in GB/s |
| [005_softmax](problems/005_softmax.py) · [Starter](shaders/starters/005_softmax.metal) · [Solution](shaders/solutions/005_softmax.metal) | Numerically stable online / multi-pass Softmax with large-offset tests | Effective bandwidth in GB/s |
| [010_2d_convolution](problems/010_2d_convolution.py) · [Starter](shaders/starters/010_2d_convolution.metal) · [Solution](shaders/solutions/010_2d_convolution.metal) | 2D valid cross-correlation / convolution with window sliding | Throughput in GFLOP/s |
| [013_histogramming](problems/013_histogramming.py) · [Starter](shaders/starters/013_histogramming.metal) · [Solution](shaders/solutions/013_histogramming.metal) | Histogram binning with hardware atomic additions | Effective bandwidth in GB/s |
| [016_prefix_sum](problems/016_prefix_sum.py) · [Starter](shaders/starters/016_prefix_sum.metal) · [Solution](shaders/solutions/016_prefix_sum.metal) | Inclusive scan prefix sum testing intra-warp and inter-threadgroup scan | Effective bandwidth in GB/s |
| [017_dot_product](problems/017_dot_product.py) · [Starter](shaders/starters/017_dot_product.metal) · [Solution](shaders/solutions/017_dot_product.metal) | Vector inner product using FMA instructions and collective reduction | Effective bandwidth in GB/s |
| [027_mean_squared_error](problems/027_mean_squared_error.py) · [Starter](shaders/starters/027_mean_squared_error.metal) · [Solution](shaders/solutions/027_mean_squared_error.metal) | Mean squared error (MSE) loss computation between prediction and target | Effective bandwidth in GB/s |
| [042_2d_max_pooling](problems/042_2d_max_pooling.py) · [Starter](shaders/starters/042_2d_max_pooling.metal) · [Solution](shaders/solutions/042_2d_max_pooling.metal) | 2D 2x2 max pooling with stride 2 for spatial downsampling | Effective bandwidth in GB/s |
| [050_rms_normalization](problems/050_rms_normalization.py) · [Starter](shaders/starters/050_rms_normalization.metal) · [Solution](shaders/solutions/050_rms_normalization.metal) | Root Mean Square Normalization layer widely used in modern LLMs | Effective bandwidth in GB/s |
| [061_rope_embedding](problems/061_rope_embedding.py) · [Starter](shaders/starters/061_rope_embedding.metal) · [Solution](shaders/solutions/061_rope_embedding.metal) | Rotary Positional Embedding (RoPE) used in LLaMA, Mistral, and Qwen | Effective bandwidth in GB/s |
| [113_layer_normalization](problems/113_layer_normalization.py) · [Starter](shaders/starters/113_layer_normalization.metal) · [Solution](shaders/solutions/113_layer_normalization.metal) | Layer Normalization (mean, variance, affine scaling) for Transformers | Effective bandwidth in GB/s |

```sh
python3 scripts/bench.py 002_matrix_multiplication --shader shaders/solutions/002_matrix_multiplication.metal
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

For an in-place problem (007 and 019), set `IN_PLACE = True` to use the single input as the output. The runner restores its snapshot before every warmup and timed dispatch, outside GPU timing. Set `REFERENCE_USES_SCALARS = True` when the CPU reference needs the constants appended after its input arrays in binding order.

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

## AI disclaimer

The Metal kernels in `shaders/solutions/` were generated with AI as examples for comparison. They can contain bugs or miss optimizations. The runner checks the bundled cases against NumPy. Test additional inputs and measure performance on your own GPU before reusing a kernel.

## Acknowledgments

Thanks to [LeetGPU](https://leetgpu.com/) for inspiring this project’s approach to learning GPU programming through kernel exercises.

## License

This project and Apple's `metal-cpp` submodule are licensed under the Apache License 2.0. See [LICENSE](LICENSE) and [metal-cpp LICENSE](third_party/metal-cpp/LICENSE.txt).
