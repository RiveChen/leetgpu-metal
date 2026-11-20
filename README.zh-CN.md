# LeetGPU in Metal

[English](README.md) · 简体中文

在 Apple Silicon Mac 上编写 Metal 计算内核，用 NumPy 验证结果，并测量 GPU 执行时间。每道题由一份 Python 题目定义和一个 `.metal` 文件组成。C++ 运行器会在运行时编译着色器，改完内核即可用同一条命令重新测试。

## 快速开始

准备一台支持 Metal 的 Apple Silicon Mac，安装带有 C++17 编译器的 Xcode Command Line Tools、CMake 3.20 或更高版本、Ninja，以及 Python 3.12 或更高版本。运行器用到的 Metal 数学模式接口从 macOS 15 开始提供。示例和测试已在 macOS 27.2.1、Python 3.15.2、NumPy 2.5.5 环境下运行通过。

克隆时一并获取子模块，其中包含 Apple 的 metal-cpp 头文件。后续命令都在仓库根目录执行。

```sh
git clone --recurse-submodules https://github.com/RiveChen/leetgpu-metal.git
cd leetgpu-metal

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install 'numpy>=2.5.5'

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
python3 scripts/bench.py 001_vector_add
```

如果已经克隆仓库，但 `third_party/metal-cpp` 目录为空，请先运行 `git submodule update --init --recursive`，再执行构建。

向量加法会运行三个用例。每个通过的用例都会输出数值误差、GPU 耗时的中位数和最小值（毫秒），以及有效带宽（GB/s）。全部通过时，最后一行是 `ALL CASES PASSED.`。要用相同的用例测试自己的实现，可以复制着色器，修改副本，再通过 `--shader` 指定它。

```sh
cp shaders/001_vector_add.metal my_vector_add.metal
python3 scripts/bench.py 001_vector_add --shader my_vector_add.metal
```

## 内置题目

| 题目 | 计算内容 | 性能指标 |
| --- | --- | --- |
| [001_vector_add](problems/001_vector_add.py) · [Metal 内核](shaders/001_vector_add.metal) | FP32 向量逐元素相加，长度分别为 31、100,003 和 16,777,216 | 有效带宽 GB/s，每个元素按两次读、一次写计算 |
| [002_matrix_mul](problems/002_matrix_mul.py) · [Metal 内核](shaders/002_matrix_mul.metal) | FP32 矩阵乘法，覆盖非方阵及最大 4096 × 4096 的方阵 | 吞吐量 GFLOP/s，运算量按 `2 × M × N × K` 计算 |

```sh
python3 scripts/bench.py 002_matrix_mul
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

创建一对同名文件 `problems/<name>.py` 和 `shaders/<name>.metal`，然后用 `python3 scripts/bench.py <name>` 运行。可以参考向量加法的两个文件。[题目定义](problems/001_vector_add.py) · [Metal 内核](shaders/001_vector_add.metal)

1. 定义 `TEST_CASES` 并实现 `make_inputs(case, rng)`，依次返回输入数组列表、输出规格和常量列表。单输出填写形状，并用 `DTYPE` 指定类型（默认 `np.float32`）。多输出填写由 `(shape, dtype)` 组成的列表。
2. 实现 `reference_impl(*inputs)`，用只读输入快照计算参考值。单输出返回一个数组，多输出返回数组列表或元组。输出形状和类型必须匹配。浮点结果必须是有限值，并满足 `RTOL` 和 `ATOL` 指定的容差（两者默认均为 `1e-3`）。整数和布尔结果逐元素精确比较。
3. 实现 `dispatch_config(case)`，返回 `(gx, gy, gz, tx, ty, tz)`。前三项是各维度的线程组数，后三项是各维度的组内线程数。如果网格尺寸按线程数计算，在末尾追加第七项 `False`。
4. 从 `[[buffer(0)]]` 开始，依次绑定输入、输出和常量。向量加法的两个输入占用槽位 0 和 1，输出占用槽位 2，长度 `np.uint32(N)` 占用槽位 3。需要显示题目特有的吞吐量时，可实现 `metrics(case, median_ms)`。

## 运行测试

构建好动态库并激活 Python 环境后，在可访问 Metal GPU 的 Mac 上运行测试。测试覆盖输出漏写、输入被改写、空浮点输出、缓冲区页尾填充和运行次数校验。

```sh
python3 -m unittest discover -s tests -v
```

## 许可证

本项目和 Apple 的 metal-cpp 子模块均采用 Apache License 2.0。[项目许可证](LICENSE) · [metal-cpp 许可证](third_party/metal-cpp/LICENSE.txt)
