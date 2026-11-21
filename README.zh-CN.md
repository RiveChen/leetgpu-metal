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

<a name="easy"></a>[Easy（20）](problems/README.zh-CN.md#easy) · <a name="medium"></a>[Medium（92）](problems/README.zh-CN.md#medium) · <a name="hard"></a>[Hard（35）](problems/README.zh-CN.md#hard)

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
2. 实现 `reference_impl(*inputs)`，用只读输入快照计算参考值。单输出返回一个数组，多输出返回数组列表或元组。输出形状和类型必须匹配。默认要求浮点结果为有限值。题目需要用无穷表示不可达等结果时，可设置 `ALLOW_EXPECTED_INF = True`，此时无穷的位置和符号必须与参考值一致。有限值需满足 `RTOL` 和 `ATOL` 指定的容差（两者默认均为 `1e-3`）。整数和布尔结果逐元素精确比较。
3. 实现 `dispatch_config(case)`，返回 `(gx, gy, gz, tx, ty, tz)`。前三项是各维度的线程组数，后三项是各维度的组内线程数。如果网格尺寸按线程数计算，在末尾追加第七项 `False`。
4. 从 `[[buffer(0)]]` 开始，依次绑定输入、输出和常量。向量加法的两个输入占用槽位 0 和 1，输出占用槽位 2，长度 `np.uint32(N)` 占用槽位 3。需要显示题目特有的吞吐量时，可实现 `metrics(case, median_ms)`。

需要临时存储的题目可用 `workspace_bytes(case)` 返回字节数。运行器将可写工作缓冲区绑定在最后一个标量之后。

原地题目（例如 015 和 022）用 `IN_PLACE = True` 将第一个输入同时绑定为输出，其他输入均为只读。每次预热或计时前，运行器从快照恢复该缓冲区，恢复操作位于 GPU 计时区间之外。当 CPU 参考值依赖标量参数时，设置 `REFERENCE_USES_SCALARS = True`，运行器会将常量按绑定顺序追加到参考函数的输入参数之后。

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
