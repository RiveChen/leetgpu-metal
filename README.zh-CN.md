# LeetGPU in Metal

[English](README.md) · 简体中文

在 Apple Silicon Mac 上编写 Metal 计算内核，用 NumPy 验证结果，并测量 GPU 执行时间。每道题都包含 Python 题目定义、Metal 练习模板和 AI 生成的参考实现。C++ 运行器会在运行时编译着色器，改完内核即可用同一条命令重新测试。

## TL;DR

在装有 Xcode Command Line Tools、CMake 和 uv 的 Apple Silicon Mac 上执行以下步骤。

```sh
git clone --recurse-submodules https://github.com/RiveChen/leetgpu-metal.git
cd leetgpu-metal
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 打开 shaders/starters/001_vector_add.metal，在 solve 的 TODO 处写入你的实现。
uv run scripts/bench.py 001_vector_add
```

## 快速开始

准备一台支持 Metal 的 Apple Silicon Mac，安装带有 C++17 编译器的 Xcode Command Line Tools、CMake 3.20 或更高版本，以及 Python 3.12 或更高版本。运行器用到的 Metal 数学模式接口从 macOS 15 开始提供。示例和测试已在 macOS 27.2.1、Python 3.15.2、NumPy 2.5.5 环境下运行通过。

克隆时一并获取子模块，其中包含 Apple 的 metal-cpp 头文件。后续命令都在仓库根目录执行。

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

如果已经克隆仓库，但 `third_party/metal-cpp` 目录为空，请先运行 `git submodule update --init --recursive`，再执行构建。

向量加法会运行三个用例。每个通过的用例都会输出数值误差、GPU 耗时的中位数和最小值（毫秒），以及有效带宽（GB/s）。全部通过时，最后一行是 `ALL CASES PASSED.`。上面的命令用参考实现验证安装。开始练习时，在 `shaders/starters/001_vector_add.metal` 的 TODO 处完成实现，再执行下面的默认命令。空模板可以编译，运行时会因输出未写入而校验失败。

```sh
python3 scripts/bench.py 001_vector_add
```

### 使用 uv

使用 Homebrew 的读者可运行 `brew install uv`。使用 uv 时，跳过上面的 venv 和 pip 步骤，照常用 CMake 构建动态库。uv 会读取 `scripts/bench.py` 声明的 Python 版本和 NumPy 依赖。先验证参考实现，再运行自己完成的模板。

```sh
uv run scripts/bench.py 001_vector_add --shader shaders/solutions/001_vector_add.metal
uv run scripts/bench.py 001_vector_add
```

## 内置题目

### Easy

题号和英文标题对应[上游 Easy 目录](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy)。本地使用独立编写的 Metal 实现和测试数据。

| 上游题目 | 题目 | 计算内容 | 性能指标 |
| --- | --- | --- | --- |
| [1. Vector Addition](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/1_vector_add) | [001_vector_add](problems/001_vector_add.py) · [练习模板](shaders/starters/001_vector_add.metal) · [参考实现](shaders/solutions/001_vector_add.metal) | FP32 向量逐元素相加，长度分别为 31、100,003 和 16,777,216 | 有效带宽 GB/s，每个元素按两次读、一次写计算 |
| [2. Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/2_matrix_multiplication) | [002_matrix_multiplication](problems/002_matrix_multiplication.py) · [练习模板](shaders/starters/002_matrix_multiplication.metal) · [参考实现](shaders/solutions/002_matrix_multiplication.metal) | FP32 矩阵乘法，覆盖非方阵及最大 4096 × 4096 的方阵 | 吞吐量 GFLOP/s，运算量按 `2 × M × N × K` 计算 |
| [3. Matrix Transpose](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/3_matrix_transpose) | [003_matrix_transpose](problems/003_matrix_transpose.py) · [练习模板](shaders/starters/003_matrix_transpose.metal) · [参考实现](shaders/solutions/003_matrix_transpose.metal) | 二维矩阵转置，测试访存合并与线程组分块优化 | 有效带宽 GB/s |
| [7. Color Inversion](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/7_color_inversion) | [007_color_inversion](problems/007_color_inversion.py) · [练习模板](shaders/starters/007_color_inversion.metal) · [参考实现](shaders/solutions/007_color_inversion.metal) | 原地反转 RGBA 图像中的 RGB，保留透明度 | GPU ms |
| [8. Matrix Addition](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/8_matrix_addition) | [008_matrix_addition](problems/008_matrix_addition.py) · [练习模板](shaders/starters/008_matrix_addition.metal) · [参考实现](shaders/solutions/008_matrix_addition.metal) | 二维矩阵逐元素加法，测试二维线程网格分发 | 有效带宽 GB/s |
| [9. 1D Convolution](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/9_1d_convolution) | [009_1d_convolution](problems/009_1d_convolution.py) · [练习模板](shaders/starters/009_1d_convolution.metal) · [参考实现](shaders/solutions/009_1d_convolution.metal) | 一维有效互相关 | GPU ms |
| [19. Reverse Array](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/19_reverse_array) | [019_reverse_array](problems/019_reverse_array.py) · [练习模板](shaders/starters/019_reverse_array.metal) · [参考实现](shaders/solutions/019_reverse_array.metal) | 原地反转 FP32 数组 | GPU ms |
| [21. ReLU](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/21_relu) | [021_relu](problems/021_relu.py) · [练习模板](shaders/starters/021_relu.metal) · [参考实现](shaders/solutions/021_relu.metal) | 逐元素单目 ReLU 激活函数，测试硬件显存带宽极限 | 有效带宽 GB/s |
| [23. Leaky ReLU](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/23_leaky_relu) | [023_leaky_relu](problems/023_leaky_relu.py) · [练习模板](shaders/starters/023_leaky_relu.metal) · [参考实现](shaders/solutions/023_leaky_relu.metal) | 对负输入使用 0.01 的斜率 | GPU ms |
| [24. Rainbow Table](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/24_rainbow_table) | [024_rainbow_table](problems/024_rainbow_table.py) · [练习模板](shaders/starters/024_rainbow_table.metal) · [参考实现](shaders/solutions/024_rainbow_table.metal) | 对四字节整数迭代计算 32 位 FNV-1a | GPU ms |
| [31. Matrix Copy](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/31_matrix_copy) | [031_matrix_copy](problems/031_matrix_copy.py) · [练习模板](shaders/starters/031_matrix_copy.metal) · [参考实现](shaders/solutions/031_matrix_copy.metal) | 复制 FP32 方阵 | GPU ms |
| [41. Simple Inference](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/41_simple_inference) | [041_simple_inference](problems/041_simple_inference.py) · [练习模板](shaders/starters/041_simple_inference.metal) · [参考实现](shaders/solutions/041_simple_inference.metal) | 批量线性层推理，以权重和偏置数组表示模型，无偏置时传零向量 | GPU ms |
| [52. Sigmoid Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/52_silu) | [052_silu](problems/052_silu.py) · [练习模板](shaders/starters/052_silu.metal) · [参考实现](shaders/solutions/052_silu.metal) | 现代 LLM 与扩散模型核心 SiLU (Swish) 激活函数 | 有效带宽 GB/s |
| [54. Swish-Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/54_swiglu) | [054_swiglu](problems/054_swiglu.py) · [练习模板](shaders/starters/054_swiglu.metal) · [参考实现](shaders/solutions/054_swiglu.metal) | 前半向量经 SiLU 激活后与后半向量相乘 | GPU ms |
| [62. Value Clipping](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/62_value_clipping) | [062_value_clipping](problems/062_value_clipping.py) · [练习模板](shaders/starters/062_value_clipping.metal) · [参考实现](shaders/solutions/062_value_clipping.metal) | 将元素截断到用例指定的上下界 | GPU ms |
| [63. Interleave Arrays](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/63_interleave) | [063_interleave](problems/063_interleave.py) · [练习模板](shaders/starters/063_interleave.metal) · [参考实现](shaders/solutions/063_interleave.metal) | 交替排列两个 FP32 数组中的元素 | GPU ms |
| [65. Gaussian Error Gated Linear Unit](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/65_geglu) | [065_geglu](problems/065_geglu.py) · [练习模板](shaders/starters/065_geglu.metal) · [参考实现](shaders/solutions/065_geglu.metal) | 用后半向量的 GELU 值对前半向量逐元素加权 | GPU ms |
| [66. RGB to Grayscale](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/66_rgb_to_grayscale) | [066_rgb_to_grayscale](problems/066_rgb_to_grayscale.py) · [练习模板](shaders/starters/066_rgb_to_grayscale.metal) · [参考实现](shaders/solutions/066_rgb_to_grayscale.metal) | 将连续存放的 RGB 像素转成灰度 | GPU ms |
| [68. Sigmoid Activation](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/68_sigmoid) | [068_sigmoid](problems/068_sigmoid.py) · [练习模板](shaders/starters/068_sigmoid.metal) · [参考实现](shaders/solutions/068_sigmoid.metal) | 逐元素计算 sigmoid | GPU ms |
| [124. GELU Activation](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/easy/124_gelu) | [124_gelu](problems/124_gelu.py) · [练习模板](shaders/starters/124_gelu.metal) · [参考实现](shaders/solutions/124_gelu.metal) | 用正态分布累积分布函数计算 GELU | GPU ms |

### 其他已收录题目

| 题目 | 计算内容 | 性能指标 |
| --- | --- | --- |
| [004_reduction](problems/004_reduction.py) · [练习模板](shaders/starters/004_reduction.metal) · [参考实现](shaders/solutions/004_reduction.metal) | 并行数组规约求和，测试 SIMD-group 原语与多级规约 | 有效带宽 GB/s |
| [005_softmax](problems/005_softmax.py) · [练习模板](shaders/starters/005_softmax.metal) · [参考实现](shaders/solutions/005_softmax.metal) | 数值稳定的 Softmax 算子，包含大偏移量防溢出测试 | 有效带宽 GB/s |
| [010_2d_convolution](problems/010_2d_convolution.py) · [练习模板](shaders/starters/010_2d_convolution.metal) · [参考实现](shaders/solutions/010_2d_convolution.metal) | 二维有效卷积 / 互相关运算，测试卷积核滑窗与共享内存复用 | 吞吐量 GFLOP/s |
| [013_histogramming](problems/013_histogramming.py) · [练习模板](shaders/starters/013_histogramming.metal) · [参考实现](shaders/solutions/013_histogramming.metal) | 直方图统计，测试硬件原子加法 `atomic_fetch_add` 与争用优化 | 有效带宽 GB/s |
| [016_prefix_sum](problems/016_prefix_sum.py) · [练习模板](shaders/starters/016_prefix_sum.metal) · [参考实现](shaders/solutions/016_prefix_sum.metal) | 前缀和扫描（Inclusive Scan），测试 SIMD 组内与多级规约扫描 | 有效带宽 GB/s |
| [017_dot_product](problems/017_dot_product.py) · [练习模板](shaders/starters/017_dot_product.metal) · [参考实现](shaders/solutions/017_dot_product.metal) | 向量内积点积，测试 FMA 指令与两级树状累加规约 | 有效带宽 GB/s |
| [027_mean_squared_error](problems/027_mean_squared_error.py) · [练习模板](shaders/starters/027_mean_squared_error.metal) · [参考实现](shaders/solutions/027_mean_squared_error.metal) | 均方误差损失（MSE Loss）计算，测试差值平方规约 | 有效带宽 GB/s |
| [042_2d_max_pooling](problems/042_2d_max_pooling.py) · [练习模板](shaders/starters/042_2d_max_pooling.metal) · [参考实现](shaders/solutions/042_2d_max_pooling.metal) | 二维 2x2 步长 2 最大池化，CNN 特征图经典降采样 | 有效带宽 GB/s |
| [050_rms_normalization](problems/050_rms_normalization.py) · [练习模板](shaders/starters/050_rms_normalization.metal) · [参考实现](shaders/solutions/050_rms_normalization.metal) | 现代大语言模型（LLM）常用的 RMSNorm 归一化层 | 有效带宽 GB/s |
| [061_rope_embedding](problems/061_rope_embedding.py) · [练习模板](shaders/starters/061_rope_embedding.metal) · [参考实现](shaders/solutions/061_rope_embedding.metal) | 旋转位置编码（RoPE），LLaMA / Mistral / Qwen 关键位置编码 | 有效带宽 GB/s |
| [113_layer_normalization](problems/113_layer_normalization.py) · [练习模板](shaders/starters/113_layer_normalization.metal) · [参考实现](shaders/solutions/113_layer_normalization.metal) | 经典 Transformer 架构标配 LayerNorm 层（均值、方差、仿射变换） | 有效带宽 GB/s |

```sh
python3 scripts/bench.py 002_matrix_multiplication --shader shaders/solutions/002_matrix_multiplication.metal
```

## 基准测试设置

每个用例默认预热 2 次，再计时 15 次。运行器通过 Metal 时间戳测量内核所在的 GPU 命令缓冲区。着色器编译、NumPy 参考值计算，以及每次调度前单独执行的 64 MiB 缓冲区写入，都在计时区间之外。比较不同内核时，请保持随机种子、数学模式和缓存设置一致。

| 参数 | 作用 |
| --- | --- |
| `--warmup N`、`--runs N` | 设置预热次数和计时次数，最小值分别为 0 和 1。 |
| `--seed N` | 设置生成输入数据时使用的随机种子，默认为 42。 |
| `--safe-math` | 使用安全数学模式编译。默认启用快速数学模式。 |
| `--no-flush` | 跳过用于减少缓存复用的 64 MiB 写入。 |
| `--entry NAME`、`--lib PATH` | 覆盖内核入口或动态库路径。入口默认取题目的 `ENTRY`，未指定时用 `solve`。动态库默认为 `build/libleetgpu_metal.dylib`。 |

## 新增题目

创建 `problems/<name>.py`，并在 `shaders/starters/` 和 `shaders/solutions/` 中各放一份同名的 `<name>.metal`，然后用 `python3 scripts/bench.py <name>` 运行。可以参考向量加法示例。[题目定义](problems/001_vector_add.py) · [练习模板](shaders/starters/001_vector_add.metal) · [参考实现](shaders/solutions/001_vector_add.metal)

1. 定义 `TEST_CASES` 并实现 `make_inputs(case, rng)`，依次返回输入数组列表、输出规格和常量列表。单输出填写形状，并用 `DTYPE` 指定类型（默认 `np.float32`）。多输出填写由 `(shape, dtype)` 组成的列表。
2. 实现 `reference_impl(*inputs)`，用只读输入快照计算参考值。单输出返回一个数组，多输出返回数组列表或元组。输出形状和类型必须匹配。浮点结果必须是有限值，并满足 `RTOL` 和 `ATOL` 指定的容差（两者默认均为 `1e-3`）。整数和布尔结果逐元素精确比较。
3. 实现 `dispatch_config(case)`，返回 `(gx, gy, gz, tx, ty, tz)`。前三项是各维度的线程组数，后三项是各维度的组内线程数。如果网格尺寸按线程数计算，在末尾追加第七项 `False`。
4. 从 `[[buffer(0)]]` 开始，依次绑定输入、输出和常量。向量加法的两个输入占用槽位 0 和 1，输出占用槽位 2，长度 `np.uint32(N)` 占用槽位 3。需要显示题目特有的吞吐量时，可实现 `metrics(case, median_ms)`。

原地题目（007、019）用 `IN_PLACE = True` 将唯一的输入同时绑定为输出。每次预热或计时前，运行器从快照恢复该缓冲区，恢复操作位于 GPU 计时区间之外。当 CPU 参考值依赖标量参数时，设置 `REFERENCE_USES_SCALARS = True`，运行器会将常量按绑定顺序追加到参考函数的输入参数之后。

## 运行测试

构建好动态库并激活 Python 环境后，在可访问 Metal GPU 的 Mac 上运行测试。测试覆盖输出漏写、输入被改写、空浮点输出、缓冲区页尾填充和运行次数校验。

```sh
python3 -m unittest discover -s tests -v
```

通过 uv 运行同一套测试：

```sh
uv run --python ">=3.12" --with "numpy>=2.5.5" python -m unittest discover -s tests -v
```

## AI 使用说明

`shaders/solutions/` 中的 Metal 参考实现由 AI 生成，供练习时对照。它们可能存在错误，也可能有进一步优化的空间。运行器用 NumPy 校验仓库自带的用例。复用内核前，请补测自己的输入，并在目标 GPU 上实测性能。

## 致谢

感谢 [LeetGPU](https://leetgpu.com/) 为本项目提供灵感，让我们把通过编写内核练习 GPU 编程的方式带到 Metal 上。

## 许可证

本项目和 Apple 的 metal-cpp 子模块均采用 Apache License 2.0。[项目许可证](LICENSE) · [metal-cpp 许可证](third_party/metal-cpp/LICENSE.txt)
