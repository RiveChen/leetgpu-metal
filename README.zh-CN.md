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

### Medium

以下 92 道题覆盖所链接上游版本的全部 Medium 题目。各题的 Metal 参数绑定和本地测试规模见练习模板及 Python 规格。第 115 题更新后的 KV 缓存和第 127 题更新后的 AdamW 状态分别写入独立的输出缓冲区，便于运行器重复执行和校验。

| 上游题目 | 题目 | 计算内容 | 性能指标 |
| --- | --- | --- | --- |
| [4. Reduction](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/4_reduction) | [004_reduction](problems/004_reduction.py) · [练习模板](shaders/starters/004_reduction.metal) · [参考实现](shaders/solutions/004_reduction.metal) | 并行数组规约求和，测试 SIMD-group 原语与多级规约 | 有效带宽 GB/s |
| [5. Softmax](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/5_softmax) | [005_softmax](problems/005_softmax.py) · [练习模板](shaders/starters/005_softmax.metal) · [参考实现](shaders/solutions/005_softmax.metal) | 数值稳定的 Softmax 算子，包含大偏移量防溢出测试 | 有效带宽 GB/s |
| [6. Softmax Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/6_softmax_attention) | [006_softmax_attention](problems/006_softmax_attention.py) · [练习模板](shaders/starters/006_softmax_attention.metal) · [参考实现](shaders/solutions/006_softmax_attention.metal) | 逐行 softmax 的缩放点积注意力 | GPU ms |
| [10. 2D Convolution](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/10_2d_convolution) | [010_2d_convolution](problems/010_2d_convolution.py) · [练习模板](shaders/starters/010_2d_convolution.metal) · [参考实现](shaders/solutions/010_2d_convolution.metal) | 二维有效卷积 / 互相关运算，测试卷积核滑窗与共享内存复用 | 吞吐量 GFLOP/s |
| [11. 3D Convolution](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/11_3d_convolution) | [011_3d_convolution](problems/011_3d_convolution.py) · [练习模板](shaders/starters/011_3d_convolution.metal) · [参考实现](shaders/solutions/011_3d_convolution.metal) | 三维有效互相关 | GPU ms |
| [13. Histogramming](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/13_histogramming) | [013_histogramming](problems/013_histogramming.py) · [练习模板](shaders/starters/013_histogramming.metal) · [参考实现](shaders/solutions/013_histogramming.metal) | 直方图统计，测试硬件原子加法 `atomic_fetch_add` 与争用优化 | 有效带宽 GB/s |
| [16. Prefix Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/16_prefix_sum) | [016_prefix_sum](problems/016_prefix_sum.py) · [练习模板](shaders/starters/016_prefix_sum.metal) · [参考实现](shaders/solutions/016_prefix_sum.metal) | 前缀和扫描（Inclusive Scan），测试 SIMD 组内与多级规约扫描 | 有效带宽 GB/s |
| [17. Dot Product](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/17_dot_product) | [017_dot_product](problems/017_dot_product.py) · [练习模板](shaders/starters/017_dot_product.metal) · [参考实现](shaders/solutions/017_dot_product.metal) | 向量内积点积，测试 FMA 指令与两级树状累加规约 | 有效带宽 GB/s |
| [18. Sparse Matrix-Vector Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/18_sparse_matrix_vector_multiplication) | [018_sparse_matrix_vector_multiplication](problems/018_sparse_matrix_vector_multiplication.py) · [练习模板](shaders/starters/018_sparse_matrix_vector_multiplication.metal) · [参考实现](shaders/solutions/018_sparse_matrix_vector_multiplication.metal) | 行主序稀疏矩阵与稠密向量相乘 | GPU ms |
| [22. General Matrix Multiplication (GEMM)](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/22_gemm) | [022_gemm](problems/022_gemm.py) · [练习模板](shaders/starters/022_gemm.metal) · [参考实现](shaders/solutions/022_gemm.metal) | 用 FP32 累加矩阵乘积，按系数与原矩阵 C 加权求和 | GPU ms |
| [25. Categorical Cross Entropy Loss](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/25_categorical_cross_entropy_loss) | [025_categorical_cross_entropy_loss](problems/025_categorical_cross_entropy_loss.py) · [练习模板](shaders/starters/025_categorical_cross_entropy_loss.metal) · [参考实现](shaders/solutions/025_categorical_cross_entropy_loss.metal) | 从 logits 和类别索引计算平均交叉熵 | GPU ms |
| [27. Mean Squared Error](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/27_mean_squared_error) | [027_mean_squared_error](problems/027_mean_squared_error.py) · [练习模板](shaders/starters/027_mean_squared_error.metal) · [参考实现](shaders/solutions/027_mean_squared_error.metal) | 均方误差损失（MSE Loss）计算，测试差值平方规约 | 有效带宽 GB/s |
| [28. Gaussian Blur](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/28_gaussian_blur) | [028_gaussian_blur](problems/028_gaussian_blur.py) · [练习模板](shaders/starters/028_gaussian_blur.metal) · [参考实现](shaders/solutions/028_gaussian_blur.metal) | 使用给定的高斯核和零填充进行模糊处理 | GPU ms |
| [29. Top K Selection](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/29_top_k_selection) | [029_top_k_selection](problems/029_top_k_selection.py) · [练习模板](shaders/starters/029_top_k_selection.metal) · [参考实现](shaders/solutions/029_top_k_selection.metal) | 取最大的 k 个值并降序排列 | GPU ms |
| [30. Batched Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/30_batched_matrix_multiplication) | [030_batched_matrix_multiplication](problems/030_batched_matrix_multiplication.py) · [练习模板](shaders/starters/030_batched_matrix_multiplication.metal) · [参考实现](shaders/solutions/030_batched_matrix_multiplication.metal) | 批次内逐对矩阵相乘 | GPU ms |
| [32. INT8 Quantized MatMul](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/32_int8_quantized_matmul) | [032_int8_quantized_matmul](problems/032_int8_quantized_matmul.py) · [练习模板](shaders/starters/032_int8_quantized_matmul.metal) · [参考实现](shaders/solutions/032_int8_quantized_matmul.metal) | 计算非对称 INT8 矩阵乘积并重新量化 | GPU ms |
| [33. Ordinary Least Squares](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/33_ordinary_least_squares) | [033_ordinary_least_squares](problems/033_ordinary_least_squares.py) · [练习模板](shaders/starters/033_ordinary_least_squares.metal) · [参考实现](shaders/solutions/033_ordinary_least_squares.metal) | 在满秩设计矩阵上拟合无截距的最小二乘模型 | GPU ms |
| [34. Logistic Regression](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/34_logistic_regression) | [034_logistic_regression](problems/034_logistic_regression.py) · [练习模板](shaders/starters/034_logistic_regression.metal) · [参考实现](shaders/solutions/034_logistic_regression.metal) | 拟合带 L2 正则项的二元逻辑回归 | GPU ms |
| [35. Monte Carlo Integration](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/35_monte_carlo_integration) | [035_monte_carlo_integration](problems/035_monte_carlo_integration.py) · [练习模板](shaders/starters/035_monte_carlo_integration.metal) · [参考实现](shaders/solutions/035_monte_carlo_integration.metal) | 用区间上的函数采样值估计定积分 | GPU ms |
| [37. Matrix Power](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/37_matrix_power) | [037_matrix_power](problems/037_matrix_power.py) · [练习模板](shaders/starters/037_matrix_power.metal) · [参考实现](shaders/solutions/037_matrix_power.metal) | 计算方阵的正整数次幂 | GPU ms |
| [38. Nearest Neighbor](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/38_nearest_neighbor) | [038_nearest_neighbor](problems/038_nearest_neighbor.py) · [练习模板](shaders/starters/038_nearest_neighbor.metal) · [参考实现](shaders/solutions/038_nearest_neighbor.metal) | 查找每个三维点的最近邻，等距时取最小索引 | GPU ms |
| [40. Batch Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/40_batch_normalization) | [040_batch_normalization](problems/040_batch_normalization.py) · [练习模板](shaders/starters/040_batch_normalization.metal) · [参考实现](shaders/solutions/040_batch_normalization.metal) | 按通道用批次统计量归一化并应用仿射变换 | GPU ms |
| [42. 2D Max Pooling](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/42_2d_max_pooling) | [042_2d_max_pooling](problems/042_2d_max_pooling.py) · [练习模板](shaders/starters/042_2d_max_pooling.metal) · [参考实现](shaders/solutions/042_2d_max_pooling.metal) | 二维 2x2 步长 2 最大池化，CNN 特征图经典降采样 | 有效带宽 GB/s |
| [43. Count Array Element](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/43_count_array_element) | [043_count_array_element](problems/043_count_array_element.py) · [练习模板](shaders/starters/043_count_array_element.metal) · [参考实现](shaders/solutions/043_count_array_element.metal) | 统计目标值在一维数组中的出现次数 | GPU ms |
| [44. Count 2D Array Element](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/44_count_2d_array_element) | [044_count_2d_array_element](problems/044_count_2d_array_element.py) · [练习模板](shaders/starters/044_count_2d_array_element.metal) · [参考实现](shaders/solutions/044_count_2d_array_element.metal) | 统计目标值在二维数组中的出现次数 | GPU ms |
| [45. Count 3D Array Element](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/45_count_3d_array_element) | [045_count_3d_array_element](problems/045_count_3d_array_element.py) · [练习模板](shaders/starters/045_count_3d_array_element.metal) · [参考实现](shaders/solutions/045_count_3d_array_element.metal) | 统计目标值在三维数组中的出现次数 | GPU ms |
| [47. Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/47_subarray_sum) | [047_subarray_sum](problems/047_subarray_sum.py) · [练习模板](shaders/starters/047_subarray_sum.metal) · [参考实现](shaders/solutions/047_subarray_sum.metal) | 求闭区间内的一维元素和 | GPU ms |
| [48. 2D Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/48_2d_subarray_sum) | [048_2d_subarray_sum](problems/048_2d_subarray_sum.py) · [练习模板](shaders/starters/048_2d_subarray_sum.metal) · [参考实现](shaders/solutions/048_2d_subarray_sum.metal) | 求指定矩形区域内的元素和 | GPU ms |
| [49. 3D Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/49_3d_subarray_sum) | [049_3d_subarray_sum](problems/049_3d_subarray_sum.py) · [练习模板](shaders/starters/049_3d_subarray_sum.metal) · [参考实现](shaders/solutions/049_3d_subarray_sum.metal) | 求指定长方体区域内的元素和 | GPU ms |
| [50. RMS Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/50_rms_normalization) | [050_rms_normalization](problems/050_rms_normalization.py) · [练习模板](shaders/starters/050_rms_normalization.metal) · [参考实现](shaders/solutions/050_rms_normalization.metal) | 现代大语言模型（LLM）常用的 RMSNorm 归一化层 | 有效带宽 GB/s |
| [51. Max Subarray Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/51_max_subarray_sum) | [051_max_subarray_sum](problems/051_max_subarray_sum.py) · [练习模板](shaders/starters/051_max_subarray_sum.metal) · [参考实现](shaders/solutions/051_max_subarray_sum.metal) | 求固定长度连续子数组的最大和 | GPU ms |
| [55. Attention with Linear Biases](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/55_attn_w_linear_bias) | [055_attn_w_linear_bias](problems/055_attn_w_linear_bias.py) · [练习模板](shaders/starters/055_attn_w_linear_bias.metal) · [参考实现](shaders/solutions/055_attn_w_linear_bias.metal) | 在注意力分数中加入与相对位置成正比的偏置 | GPU ms |
| [57. FP16 Batched Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/57_fp16_batched_matmul) | [057_fp16_batched_matmul](problems/057_fp16_batched_matmul.py) · [练习模板](shaders/starters/057_fp16_batched_matmul.metal) · [参考实现](shaders/solutions/057_fp16_batched_matmul.metal) | 批量计算 FP16 矩阵乘积，使用 FP32 累加 | GPU ms |
| [58. FP16 Dot Product](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/58_fp16_dot_product) | [058_fp16_dot_product](problems/058_fp16_dot_product.py) · [练习模板](shaders/starters/058_fp16_dot_product.metal) · [参考实现](shaders/solutions/058_fp16_dot_product.metal) | 用 FP32 累加向量内积并写回 FP16 结果 | GPU ms |
| [60. Top-p Sampling](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/60_top_p_sampling) | [060_top_p_sampling](problems/060_top_p_sampling.py) · [练习模板](shaders/starters/060_top_p_sampling.metal) · [参考实现](shaders/solutions/060_top_p_sampling.metal) | 在累计概率达到 p 的候选集中按种子采样 | GPU ms |
| [61. Rotary Positional Embedding](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/61_rope_embedding) | [061_rope_embedding](problems/061_rope_embedding.py) · [练习模板](shaders/starters/061_rope_embedding.metal) · [参考实现](shaders/solutions/061_rope_embedding.metal) | 旋转位置编码（RoPE），LLaMA / Mistral / Qwen 关键位置编码 | 有效带宽 GB/s |
| [64. Weight Dequantization](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/64_weight_dequantization) | [064_weight_dequantization](problems/064_weight_dequantization.py) · [练习模板](shaders/starters/064_weight_dequantization.metal) · [参考实现](shaders/solutions/064_weight_dequantization.metal) | 按矩阵块对应的缩放系数反量化权重 | GPU ms |
| [67. MoE Top-K Gating](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/67_moe_topk_gating) | [067_moe_topk_gating](problems/067_moe_topk_gating.py) · [练习模板](shaders/starters/067_moe_topk_gating.metal) · [参考实现](shaders/solutions/067_moe_topk_gating.metal) | 为每个 token 选择前 k 个专家并归一化路由权重 | GPU ms |
| [69. 2D Jacobi Stencil](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/69_jacobi_stencil_2d) | [069_jacobi_stencil_2d](problems/069_jacobi_stencil_2d.py) · [练习模板](shaders/starters/069_jacobi_stencil_2d.metal) · [参考实现](shaders/solutions/069_jacobi_stencil_2d.metal) | 保留边界，使用上下左右四邻域执行 Jacobi 更新 | GPU ms |
| [70. Segmented Exclusive Prefix Sum](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/70_segmented_prefix_sum) | [070_segmented_prefix_sum](problems/070_segmented_prefix_sum.py) · [练习模板](shaders/starters/070_segmented_prefix_sum.metal) · [参考实现](shaders/solutions/070_segmented_prefix_sum.metal) | 按段计算不含当前位置的前缀和 | GPU ms |
| [71. Parallel Merge](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/71_parallel_merge) | [071_parallel_merge](problems/071_parallel_merge.py) · [练习模板](shaders/starters/071_parallel_merge.metal) · [参考实现](shaders/solutions/071_parallel_merge.metal) | 合并两个有序数组并保留重复元素 | GPU ms |
| [72. Stream Compaction](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/72_stream_compaction) | [072_stream_compaction](problems/072_stream_compaction.py) · [练习模板](shaders/starters/072_stream_compaction.metal) · [参考实现](shaders/solutions/072_stream_compaction.metal) | 按原顺序压紧正数并将末尾补零 | GPU ms |
| [75. Sparse Matrix-Dense Matrix Multiplication](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/75_sparse_matrix_dense_matrix_multiplication) | [075_sparse_matrix_dense_matrix_multiplication](problems/075_sparse_matrix_dense_matrix_multiplication.py) · [练习模板](shaders/starters/075_sparse_matrix_dense_matrix_multiplication.metal) · [参考实现](shaders/solutions/075_sparse_matrix_dense_matrix_multiplication.metal) | 将行主序存储的稀疏矩阵与稠密矩阵相乘 | GPU ms |
| [76. Adder Transformer Inference](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/76_adder_transformer) | [076_adder_transformer](problems/076_adder_transformer.py) · [练习模板](shaders/starters/076_adder_transformer.metal) · [参考实现](shaders/solutions/076_adder_transformer.metal) | 用加法 Transformer 贪心解码 11 步，输出每步的数字分数 | GPU ms |
| [77. Integral Image](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/77_integral_image) | [077_integral_image](problems/077_integral_image.py) · [练习模板](shaders/starters/077_integral_image.metal) · [参考实现](shaders/solutions/077_integral_image.metal) | 计算每个像素对应的二维前缀和 | GPU ms |
| [78. 2D FFT](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/78_2d_fft) | [078_2d_fft](problems/078_2d_fft.py) · [练习模板](shaders/starters/078_2d_fft.metal) · [参考实现](shaders/solutions/078_2d_fft.metal) | 对交错存储的复数矩阵执行二维离散傅里叶变换 | GPU ms |
| [79. Connected Components](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/79_connected_components) | [079_connected_components](problems/079_connected_components.py) · [练习模板](shaders/starters/079_connected_components.metal) · [参考实现](shaders/solutions/079_connected_components.metal) | 用连通分量中的最小顶点编号标记无向图顶点 | GPU ms |
| [80. Grouped Query Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/80_grouped_query_attention) | [080_grouped_query_attention](problems/080_grouped_query_attention.py) · [练习模板](shaders/starters/080_grouped_query_attention.metal) · [参考实现](shaders/solutions/080_grouped_query_attention.metal) | 在分组查询注意力中让多组查询头共享键值头 | GPU ms |
| [81. INT4 Weight-Only Quantized MatMul](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/81_int4_matmul) | [081_int4_matmul](problems/081_int4_matmul.py) · [练习模板](shaders/starters/081_int4_matmul.metal) · [参考实现](shaders/solutions/081_int4_matmul.metal) | 解包 INT4 权重并按组缩放，与 FP16 激活相乘 | GPU ms |
| [82. Linear Recurrence](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/82_linear_recurrence) | [082_linear_recurrence](problems/082_linear_recurrence.py) · [练习模板](shaders/starters/082_linear_recurrence.metal) · [参考实现](shaders/solutions/082_linear_recurrence.metal) | 逐序列计算带时变系数的一阶线性递推 | GPU ms |
| [83. Fused Residual Add and RMS Norm](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/83_fused_residual_add_rms_norm) | [083_fused_residual_add_rms_norm](problems/083_fused_residual_add_rms_norm.py) · [练习模板](shaders/starters/083_fused_residual_add_rms_norm.metal) · [参考实现](shaders/solutions/083_fused_residual_add_rms_norm.metal) | 将残差相加，再逐行执行加权 RMS 归一化 | GPU ms |
| [84. SwiGLU MLP Block](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/84_swiglu_mlp_block) | [084_swiglu_mlp_block](problems/084_swiglu_mlp_block.py) · [练习模板](shaders/starters/084_swiglu_mlp_block.metal) · [参考实现](shaders/solutions/084_swiglu_mlp_block.metal) | 计算门控投影、SiLU 激活和输出投影 | GPU ms |
| [85. LoRA Linear](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/85_lora_linear) | [085_lora_linear](problems/085_lora_linear.py) · [练习模板](shaders/starters/085_lora_linear.metal) · [参考实现](shaders/solutions/085_lora_linear.metal) | 将缩放后的低秩适配器输出叠加到基础线性层 | GPU ms |
| [86. Paged KV-Cache Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/86_paged_attention) | [086_paged_attention](problems/086_paged_attention.py) · [练习模板](shaders/starters/086_paged_attention.metal) · [参考实现](shaders/solutions/086_paged_attention.metal) | 通过块表读取有效的键值缓存并计算注意力 | GPU ms |
| [87. Speculative Decoding Verification](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/87_speculative_decoding_verification) | [087_speculative_decoding_verification](problems/087_speculative_decoding_verification.py) · [练习模板](shaders/starters/087_speculative_decoding_verification.metal) · [参考实现](shaders/solutions/087_speculative_decoding_verification.metal) | 验证草稿 token，拒绝时采样替代项，全部接受时额外采样 | GPU ms |
| [88. Shared-Prefix Cascade Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/88_shared_prefix_attention) | [088_shared_prefix_attention](problems/088_shared_prefix_attention.py) · [练习模板](shaders/starters/088_shared_prefix_attention.metal) · [参考实现](shaders/solutions/088_shared_prefix_attention.metal) | 对共享前缀及各自后缀的可见部分计算注意力 | GPU ms |
| [90. Causal Depthwise Conv1d](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/90_causal_depthwise_conv1d) | [090_causal_depthwise_conv1d](problems/090_causal_depthwise_conv1d.py) · [练习模板](shaders/starters/090_causal_depthwise_conv1d.metal) · [参考实现](shaders/solutions/090_causal_depthwise_conv1d.metal) | 对各通道独立计算一维因果卷积 | GPU ms |
| [91. MoE Token Dispatch](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/91_moe_token_dispatch) | [091_moe_token_dispatch](problems/091_moe_token_dispatch.py) · [练习模板](shaders/starters/091_moe_token_dispatch.metal) · [参考实现](shaders/solutions/091_moe_token_dispatch.metal) | 按专家分发 token 并记录每个专家收到的数量 | GPU ms |
| [92. Decaying Causal Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/92_decaying_causal_attention) | [092_decaying_causal_attention](problems/092_decaying_causal_attention.py) · [练习模板](shaders/starters/092_decaying_causal_attention.metal) · [参考实现](shaders/solutions/092_decaying_causal_attention.metal) | 用随时间衰减的权重累加因果注意力 | GPU ms |
| [94. SSM Selective Scan](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/94_ssm_selective_scan) | [094_ssm_selective_scan](problems/094_ssm_selective_scan.py) · [练习模板](shaders/starters/094_ssm_selective_scan.metal) · [参考实现](shaders/solutions/094_ssm_selective_scan.metal) | 沿时间维扫描选择性状态空间模型并加入跳跃连接 | GPU ms |
| [96. INT8 KV-Cache Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/96_int8_kv_cache_attention) | [096_int8_kv_cache_attention](problems/096_int8_kv_cache_attention.py) · [练习模板](shaders/starters/096_int8_kv_cache_attention.metal) · [参考实现](shaders/solutions/096_int8_kv_cache_attention.metal) | 按 token 反量化 INT8 键值缓存并分头计算注意力 | GPU ms |
| [98. Beam Search Step](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/98_beam_search_step) | [098_beam_search_step](problems/098_beam_search_step.py) · [练习模板](shaders/starters/098_beam_search_step.metal) · [参考实现](shaders/solutions/098_beam_search_step.metal) | 选择得分最高的下一步候选及其父束索引 | GPU ms |
| [99. Grouped GEMM](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/99_grouped_gemm) | [099_grouped_gemm](problems/099_grouped_gemm.py) · [练习模板](shaders/starters/099_grouped_gemm.metal) · [参考实现](shaders/solutions/099_grouped_gemm.metal) | 按行偏移分组并分别计算矩阵乘积 | GPU ms |
| [101. N-body Gravitational Force](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/101_n_body_force) | [101_n_body_force](problems/101_n_body_force.py) · [练习模板](shaders/starters/101_n_body_force.metal) · [参考实现](shaders/solutions/101_n_body_force.metal) | 累加带软化项的三维多体引力 | GPU ms |
| [102. Variable-Length Causal Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/102_varlen_causal_attention) | [102_varlen_causal_attention](problems/102_varlen_causal_attention.py) · [练习模板](shaders/starters/102_varlen_causal_attention.metal) · [参考实现](shaders/solutions/102_varlen_causal_attention.metal) | 对紧凑存储的变长序列分别计算因果注意力 | GPU ms |
| [103. Hadamard Transform](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/103_hadamard_transform) | [103_hadamard_transform](problems/103_hadamard_transform.py) · [练习模板](shaders/starters/103_hadamard_transform.metal) · [参考实现](shaders/solutions/103_hadamard_transform.metal) | 对每行执行正交归一化的 Walsh–Hadamard 变换 | GPU ms |
| [104. Min-P Sampling](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/104_min_p_sampling) | [104_min_p_sampling](problems/104_min_p_sampling.py) · [练习模板](shaders/starters/104_min_p_sampling.metal) · [参考实现](shaders/solutions/104_min_p_sampling.metal) | 按最大概率的指定比例筛选 token 并重新归一化 | GPU ms |
| [105. Group Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/105_group_normalization) | [105_group_normalization](problems/105_group_normalization.py) · [练习模板](shaders/starters/105_group_normalization.metal) · [参考实现](shaders/solutions/105_group_normalization.metal) | 在 NCHW 张量上按组归一化并逐通道作仿射变换 | GPU ms |
| [106. Token Embedding Layer](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/106_token_embedding_layer) | [106_token_embedding_layer](problems/106_token_embedding_layer.py) · [练习模板](shaders/starters/106_token_embedding_layer.metal) · [参考实现](shaders/solutions/106_token_embedding_layer.metal) | 叠加 token 和位置嵌入，再逐 token 作层归一化 | GPU ms |
| [107. PPO Clipped Surrogate Loss](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/107_ppo_clipped_surrogate_loss) | [107_ppo_clipped_surrogate_loss](problems/107_ppo_clipped_surrogate_loss.py) · [练习模板](shaders/starters/107_ppo_clipped_surrogate_loss.metal) · [参考实现](shaders/solutions/107_ppo_clipped_surrogate_loss.metal) | 计算 PPO 裁剪策略目标的负均值 | GPU ms |
| [108. DPO Sequence Loss](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/108_dpo_sequence_loss) | [108_dpo_sequence_loss](problems/108_dpo_sequence_loss.py) · [练习模板](shaders/starters/108_dpo_sequence_loss.metal) · [参考实现](shaders/solutions/108_dpo_sequence_loss.metal) | 根据策略模型和参考模型的序列对数概率计算 DPO 损失 | GPU ms |
| [109. GRPO Surrogate Loss](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/109_grpo_surrogate_loss) | [109_grpo_surrogate_loss](problems/109_grpo_surrogate_loss.py) · [练习模板](shaders/starters/109_grpo_surrogate_loss.metal) · [参考实现](shaders/solutions/109_grpo_surrogate_loss.metal) | 用组内归一化奖励、比率裁剪和 KL 惩罚计算 GRPO 损失 | GPU ms |
| [110. Parallel Reverse Scan (GAE)](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/110_gae_reverse_scan) | [110_gae_reverse_scan](problems/110_gae_reverse_scan.py) · [练习模板](shaders/starters/110_gae_reverse_scan.metal) · [参考实现](shaders/solutions/110_gae_reverse_scan.metal) | 逆序扫描计算广义优势估计 | GPU ms |
| [111. Softmax Attention Backward](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/111_softmax_attention_backward) | [111_softmax_attention_backward](problems/111_softmax_attention_backward.py) · [练习模板](shaders/starters/111_softmax_attention_backward.metal) · [参考实现](shaders/solutions/111_softmax_attention_backward.metal) | 计算缩放点积注意力对 Q、K、V 的梯度 | GPU ms |
| [112. Attention with Sinks](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/112_attention_with_sinks) | [112_attention_with_sinks](problems/112_attention_with_sinks.py) · [练习模板](shaders/starters/112_attention_with_sinks.metal) · [参考实现](shaders/solutions/112_attention_with_sinks.metal) | 在因果窗口及固定保留的起始 token 上计算注意力 | GPU ms |
| [113. Layer Normalization](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/113_layer_normalization) | [113_layer_normalization](problems/113_layer_normalization.py) · [练习模板](shaders/starters/113_layer_normalization.metal) · [参考实现](shaders/solutions/113_layer_normalization.metal) | 经典 Transformer 架构标配 LayerNorm 层（均值、方差、仿射变换） | 有效带宽 GB/s |
| [115. Fused QKV Projection with RoPE and KV Cache Update](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/115_fused_qkv_rope_kv_cache_update) | [115_fused_qkv_rope_kv_cache_update](problems/115_fused_qkv_rope_kv_cache_update.py) · [练习模板](shaders/starters/115_fused_qkv_rope_kv_cache_update.metal) · [参考实现](shaders/solutions/115_fused_qkv_rope_kv_cache_update.metal) | 投影 Q、K、V，对 Q、K 作旋转位置编码并更新键值缓存 | GPU ms |
| [118. Vision Transformer Patch Embedding](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/118_vit_patch_embedding) | [118_vit_patch_embedding](problems/118_vit_patch_embedding.py) · [练习模板](shaders/starters/118_vit_patch_embedding.metal) · [参考实现](shaders/solutions/118_vit_patch_embedding.metal) | 投影互不重叠的图像块，加入分类 token 和位置嵌入 | GPU ms |
| [120. Kimi Delta Attention](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/120_kimi_delta_attention) | [120_kimi_delta_attention](problems/120_kimi_delta_attention.py) · [练习模板](shaders/starters/120_kimi_delta_attention.metal) · [参考实现](shaders/solutions/120_kimi_delta_attention.metal) | 按通道衰减并更新记忆矩阵，计算每步的查询输出 | GPU ms |
| [121. Fused Logit Penalties](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/121_fused_logit_penalties) | [121_fused_logit_penalties](problems/121_fused_logit_penalties.py) · [练习模板](shaders/starters/121_fused_logit_penalties.metal) · [参考实现](shaders/solutions/121_fused_logit_penalties.metal) | 按 token 施加重复、频率和存在惩罚 | GPU ms |
| [122. Hamming Top-K Search](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/122_hamming_top_k) | [122_hamming_top_k](problems/122_hamming_top_k.py) · [练习模板](shaders/starters/122_hamming_top_k.metal) · [参考实现](shaders/solutions/122_hamming_top_k.metal) | 按汉明距离选出最近的 k 个 256 位编码 | GPU ms |
| [125. Scatter Add](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/125_scatter_add) | [125_scatter_add](problems/125_scatter_add.py) · [练习模板](shaders/starters/125_scatter_add.metal) · [参考实现](shaders/solutions/125_scatter_add.metal) | 按目标行索引累加输入，支持重复索引 | GPU ms |
| [126. Row-wise Argmax](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/126_row_argmax) | [126_row_argmax](problems/126_row_argmax.py) · [练习模板](shaders/starters/126_row_argmax.metal) · [参考实现](shaders/solutions/126_row_argmax.metal) | 找出每行最大值首次出现的位置 | GPU ms |
| [127. Fused AdamW](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/127_fused_adamw) | [127_fused_adamw](problems/127_fused_adamw.py) · [练习模板](shaders/starters/127_fused_adamw.metal) · [参考实现](shaders/solutions/127_fused_adamw.metal) | 完成一步 AdamW 更新并输出参数与两组矩估计 | GPU ms |
| [128. Split-KV Attention Reduction](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/128_split_kv_attention_reduction) | [128_split_kv_attention_reduction](problems/128_split_kv_attention_reduction.py) · [练习模板](shaders/starters/128_split_kv_attention_reduction.metal) · [参考实现](shaders/solutions/128_split_kv_attention_reduction.metal) | 用稳定的对数求和权重合并分块注意力结果 | GPU ms |
| [131. Fused LayerNorm Linear GELU](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/131_fused_layernorm_linear_gelu) | [131_fused_layernorm_linear_gelu](problems/131_fused_layernorm_linear_gelu.py) · [练习模板](shaders/starters/131_fused_layernorm_linear_gelu.metal) · [参考实现](shaders/solutions/131_fused_layernorm_linear_gelu.metal) | 融合层归一化、带偏置的线性投影和 GELU | GPU ms |
| [133. Multi-LoRA Batched Linear](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/133_multi_lora_batched_linear) | [133_multi_lora_batched_linear](problems/133_multi_lora_batched_linear.py) · [练习模板](shaders/starters/133_multi_lora_batched_linear.metal) · [参考实现](shaders/solutions/133_multi_lora_batched_linear.metal) | 为每个 token 的基础投影叠加所选 LoRA 适配器 | GPU ms |
| [142. RMS Norm Backward](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/142_rms_norm_backward) | [142_rms_norm_backward](problems/142_rms_norm_backward.py) · [练习模板](shaders/starters/142_rms_norm_backward.metal) · [参考实现](shaders/solutions/142_rms_norm_backward.metal) | 计算 RMS 归一化对输入及缩放权重的梯度 | GPU ms |
| [144. Layer Norm Backward](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/144_layer_norm_backward) | [144_layer_norm_backward](problems/144_layer_norm_backward.py) · [练习模板](shaders/starters/144_layer_norm_backward.metal) · [参考实现](shaders/solutions/144_layer_norm_backward.metal) | 利用行统计量计算层归一化的输入、缩放及偏置梯度 | GPU ms |
| [150. Distillation KL Divergence](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/150_distillation_kl_divergence) | [150_distillation_kl_divergence](problems/150_distillation_kl_divergence.py) · [练习模板](shaders/starters/150_distillation_kl_divergence.metal) · [参考实现](shaders/solutions/150_distillation_kl_divergence.metal) | 计算温度缩放后的教师分布到学生分布的逐行 KL 散度 | GPU ms |
| [151. Muon Optimizer](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/151_muon_optimizer) | [151_muon_optimizer](problems/151_muon_optimizer.py) · [练习模板](shaders/starters/151_muon_optimizer.metal) · [参考实现](shaders/solutions/151_muon_optimizer.metal) | 对动量调整后的梯度执行五次 Newton–Schulz 迭代 | GPU ms |
| [155. MoE Expert Weight Gradient](https://github.com/AlphaGPU/leetgpu-challenges/tree/3b2ec31a1e088d6b8ed81962fbfd85bb92151b63/challenges/medium/155_moe_expert_weight_gradient) | [155_moe_expert_weight_gradient](problems/155_moe_expert_weight_gradient.py) · [练习模板](shaders/starters/155_moe_expert_weight_gradient.metal) · [参考实现](shaders/solutions/155_moe_expert_weight_gradient.metal) | 按专家汇总分配到的 token 对专家权重的梯度 | GPU ms |

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

需要临时存储的题目可用 `workspace_bytes(case)` 返回字节数。运行器将可写工作缓冲区绑定在最后一个标量之后。

原地题目（007、019、022）用 `IN_PLACE = True` 将第一个输入同时绑定为输出，其他输入均为只读。每次预热或计时前，运行器从快照恢复该缓冲区，恢复操作位于 GPU 计时区间之外。当 CPU 参考值依赖标量参数时，设置 `REFERENCE_USES_SCALARS = True`，运行器会将常量按绑定顺序追加到参考函数的输入参数之后。

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
