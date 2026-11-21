# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.5.5",
# ]
# ///

import argparse
import ctypes
import importlib.util
import mmap
import sys
from pathlib import Path

import numpy as np

# --- Ctypes C ABI struct definitions (strictly aligned with src/harness.hpp) ---


class BufferDesc(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("byte_length", ctypes.c_size_t),
        ("index", ctypes.c_uint32),
    ]


class BytesDesc(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("byte_length", ctypes.c_size_t),
        ("index", ctypes.c_uint32),
    ]


class DispatchConfig(ctypes.Structure):
    _fields_ = [
        ("grid_x", ctypes.c_uint32),
        ("grid_y", ctypes.c_uint32),
        ("grid_z", ctypes.c_uint32),
        ("tg_x", ctypes.c_uint32),
        ("tg_y", ctypes.c_uint32),
        ("tg_z", ctypes.c_uint32),
        ("use_threadgroups", ctypes.c_bool),
    ]


class RunResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("median_ms", ctypes.c_double),
        ("min_ms", ctypes.c_double),
        ("error_msg", ctypes.c_char * 2048),
    ]


def load_harness_lib(lib_path: Path):
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Dynamic library not found at {lib_path}. Run `cmake -S . -B build && cmake --build build` first."
        )
    lib = ctypes.CDLL(str(lib_path))
    lib.run_metal_kernel.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(BufferDesc),
        ctypes.c_size_t,
        ctypes.POINTER(BytesDesc),
        ctypes.c_size_t,
        DispatchConfig,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    lib.run_metal_kernel.restype = RunResult
    if hasattr(lib, "run_metal_kernel_inplace"):
        lib.run_metal_kernel_inplace.argtypes = [
            *lib.run_metal_kernel.argtypes,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.run_metal_kernel_inplace.restype = RunResult
    return lib


def load_problem_module(problem_path: Path):
    if not problem_path.exists():
        raise FileNotFoundError(f"Problem definition not found: {problem_path}")
    spec = importlib.util.spec_from_file_location(problem_path.stem, str(problem_path))
    if spec is None:
        raise ImportError(f"Failed to load problem module: {problem_path}")
    mod = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Failed to load problem module: {problem_path}")
    spec.loader.exec_module(mod)
    return mod


# --- Page-aligned UMA memory allocator ---

PAGE_SIZE = mmap.PAGESIZE  # Apple Silicon 自动为 16384 (16KB)，x86_64 自动为 4096 (4KB)
PADDING_CANARY_BYTE = b"\xcc"


def alloc_page_aligned(shape, dtype=np.float32, fill_value=None):
    """Allocate a page-aligned array with sentinel bytes after its payload."""
    dtype = np.dtype(dtype)
    count = int(np.prod(shape))
    payload_nbytes = count * dtype.itemsize
    # Use the next page boundary even when the payload ends exactly on a boundary.
    aligned_bytes = (payload_nbytes // PAGE_SIZE + 1) * PAGE_SIZE

    mm = mmap.mmap(-1, aligned_bytes)
    mm[:] = PADDING_CANARY_BYTE * aligned_bytes

    arr = np.frombuffer(mm, dtype=dtype, count=count).reshape(shape)
    if fill_value is not None:
        arr.fill(fill_value)
    return arr, mm, payload_nbytes


def to_aligned_copy(src: np.ndarray):
    arr, mm, payload_nbytes = alloc_page_aligned(src.shape, src.dtype)
    np.copyto(arr, np.ascontiguousarray(src))
    return arr, mm, payload_nbytes


def check_page_padding(mm: mmap.mmap, payload_nbytes: int) -> tuple[bool, str]:
    """check if the [payload_nbytes : len(mm)] range was out-of-bounds modified by the GPU kernel"""
    pad_len = len(mm) - payload_nbytes
    if pad_len <= 0:
        return True, ""
    actual_pad = bytes(mm[payload_nbytes:])
    expected_pad = PADDING_CANARY_BYTE * pad_len
    if actual_pad != expected_pad:
        # locate the byte offset of the first byte that was out-of-bounds modified
        for offset, (a, e) in enumerate(zip(actual_pad, expected_pad, strict=True)):
            if a != e:
                return False, (
                    f"Page padding overwritten at byte offset +{offset} past buffer end "
                    f"(got 0x{a:02X}, expected 0xCC)"
                )
    return True, ""


# --- Numerical precision and error diagnostics ---


def output_spec_error(out: np.ndarray, ref: np.ndarray) -> str | None:
    if out.shape != ref.shape:
        return f"Shape mismatch: got {out.shape}, expected {ref.shape}"
    if out.dtype != ref.dtype:
        return f"Dtype mismatch: got {out.dtype}, expected {ref.dtype}"
    return None


def verify_output(out: np.ndarray, ref: np.ndarray, rtol: float, atol: float):
    if error := output_spec_error(out, ref):
        return False, error

    if np.issubdtype(out.dtype, np.floating):
        if out.size == 0:
            return True, "empty output"
        if np.isnan(out).any() or np.isinf(out).any():
            nan_cnt = int(np.isnan(out).sum())
            inf_cnt = int(np.isinf(out).sum())
            return (
                False,
                f"NaN({nan_cnt})/Inf({inf_cnt}) detected (uninitialized or out-of-bounds write)",
            )

        abs_diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
        max_abs = float(abs_diff.max())
        rmse = float(np.sqrt(np.mean(abs_diff**2)))
        passed = np.allclose(out, ref, rtol=rtol, atol=atol)

        if not passed:
            idx = np.unravel_index(np.argmax(abs_diff), out.shape)
            return False, (
                f"max_abs={max_abs:.4e}, rmse={rmse:.4e} "
                f"at idx={idx}: got={out[idx]}, expected={ref[idx]}"
            )
        return True, f"max_abs={max_abs:.2e}, rmse={rmse:.2e}"
    else:
        passed = np.array_equal(out, ref)
        if not passed:
            diff_mask = out != ref
            first_idx = np.unravel_index(np.argmax(diff_mask), out.shape)
            return (
                False,
                f"Exact mismatch at idx={first_idx}: got={out[first_idx]}, expected={ref[first_idx]}",
            )
        return True, "exact match"


# --- Single test case executor ---


def run_count_error(warmup: int, runs: int) -> str | None:
    max_count = np.iinfo(np.intc).max  # Both counts cross the ABI as C ints.
    if not 0 <= warmup <= max_count:
        return f"--warmup must be between 0 and {max_count}"
    if not 1 <= runs <= max_count:
        return f"--runs must be between 1 and {max_count}"
    return None


def run_single_case(
    lib,
    mod,
    msl_bytes: bytes,
    entry_name: bytes,
    case: dict,
    rng: np.random.Generator,
    warmup: int,
    runs: int,
    flush_l2: bool,
    fast_math: bool,
):
    if error := run_count_error(warmup, runs):
        return False, error, None
    inputs, out_spec, scalars = mod.make_inputs(case, rng)

    # support single output (shape,) or multiple outputs [(shape1, dtype1), ...]
    if isinstance(out_spec, tuple) and (len(out_spec) == 0 or isinstance(out_spec[0], int)):
        out_specs = [(out_spec, getattr(mod, "DTYPE", np.float32))]
        single_output = True
    else:
        out_specs = list(out_spec)
        single_output = False

    in_place = getattr(mod, "IN_PLACE", False)
    if in_place and not hasattr(lib, "run_metal_kernel_inplace"):
        return (
            False,
            "In-place execution requires an updated library. Run cmake --build build.",
            None,
        )
    if in_place and (len(inputs) != 1 or not single_output):
        raise ValueError("IN_PLACE requires one input and one output of the same shape and dtype")
    keep_alive_mmaps = []
    buffer_descs = []
    all_buffer_guards = []  # 记录 (label, mm, payload_nbytes) 用于执行后统一审计
    slot_idx = 0

    # bind input Buffers, and keep a readonly baseline snapshot to prevent GPU from mutating inputs
    readonly_inputs = []
    input_snapshots = []
    for i, inp in enumerate(inputs):
        arr, mm, payload_nbytes = to_aligned_copy(inp)
        keep_alive_mmaps.append(mm)
        buffer_descs.append(BufferDesc(arr.ctypes.data, len(mm), slot_idx))
        all_buffer_guards.append((f"input[{i}]", mm, payload_nbytes))
        slot_idx += 1

        input_bytes = arr.reshape(-1).view(np.uint8)
        snapshot = input_bytes.copy()
        snapshot.flags.writeable = False
        readonly_inputs.append(snapshot.view(arr.dtype).reshape(arr.shape))
        if not in_place:
            input_snapshots.append((input_bytes, snapshot))
        else:
            inplace_array = arr

    # 2. must compute true values based on pure inputs before GPU execution to prevent UMA shared memory from being mutated by the Kernel
    reference_scalars = scalars if getattr(mod, "REFERENCE_USES_SCALARS", False) else ()
    raw_refs = mod.reference_impl(*readonly_inputs, *reference_scalars)
    if single_output:
        refs = [raw_refs]
    else:
        refs = list(raw_refs) if isinstance(raw_refs, (list, tuple)) else [raw_refs]

    # 3. explicitly validate output tensor count, prevent zip from silently truncating
    if len(out_specs) != len(refs):
        return (
            False,
            f"Output count mismatch: declared {len(out_specs)} outputs, but reference_impl returned {len(refs)}",
            None,
        )

    # bind output Buffers (initialized with sentinel values)
    aligned_outputs = []
    for i, (shape, dtype) in enumerate(out_specs):
        if in_place:
            if inplace_array.shape != shape or inplace_array.dtype != dtype:
                raise ValueError("In-place output must match the input buffer")
            aligned_outputs.append(inplace_array)
            continue
        arr, mm, payload_nbytes = alloc_page_aligned(shape, dtype)
        ref = np.asarray(refs[i])
        if error := output_spec_error(arr, ref):
            return False, f"FAIL (out[{i}] {error})", None
        if np.issubdtype(arr.dtype, np.floating):
            arr.fill(np.nan)
        elif arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.integer):
            # Every unwritten element must differ from its expected value.
            np.invert(ref, out=arr)
        else:
            raise TypeError(f"Unsupported output dtype for canary initialization: {arr.dtype}")
        aligned_outputs.append(arr)
        keep_alive_mmaps.append(mm)
        buffer_descs.append(BufferDesc(arr.ctypes.data, len(mm), slot_idx))
        all_buffer_guards.append((f"output[{i}]", mm, payload_nbytes))
        slot_idx += 1

    # bind scalar/structure constants
    keep_alive_scalars = []
    bytes_descs = []
    for sc in scalars:
        if isinstance(sc, np.generic):
            raw = sc.tobytes()
            buf = ctypes.create_string_buffer(raw, len(raw))
            keep_alive_scalars.append(buf)
            bytes_descs.append(BytesDesc(ctypes.addressof(buf), len(raw), slot_idx))
        elif isinstance(sc, ctypes.Structure):
            keep_alive_scalars.append(sc)
            bytes_descs.append(BytesDesc(ctypes.addressof(sc), ctypes.sizeof(sc), slot_idx))
        else:
            raise TypeError(f"Unsupported scalar type: {type(sc)}")
        slot_idx += 1

    cfg = mod.dispatch_config(case)
    if len(cfg) == 6:
        gx, gy, gz, tx, ty, tz = cfg
        use_tg = True
    elif len(cfg) == 7:
        gx, gy, gz, tx, ty, tz, use_tg = cfg
    else:
        raise ValueError(f"dispatch_config must return 6 or 7 elements, got {len(cfg)}")

    dispatch = DispatchConfig(gx, gy, gz, tx, ty, tz, bool(use_tg))

    c_bufs = (BufferDesc * len(buffer_descs))(*buffer_descs)
    c_bytes = (BytesDesc * len(bytes_descs))(*bytes_descs) if bytes_descs else None

    run_kernel = lib.run_metal_kernel_inplace if in_place else lib.run_metal_kernel
    reset_args = (readonly_inputs[0].ctypes.data, readonly_inputs[0].nbytes) if in_place else ()
    res = run_kernel(
        msl_bytes,
        entry_name,
        c_bufs,
        len(buffer_descs),
        c_bytes,
        len(bytes_descs),
        dispatch,
        warmup,
        runs,
        flush_l2,
        fast_math,
        *reset_args,
    )

    if res.status != 0:
        return False, f"GPU ERROR ({res.status}): {res.error_msg.decode()}", None

    for label, mm, payload_nbytes in all_buffer_guards:
        pad_ok, pad_msg = check_page_padding(mm, payload_nbytes)
        if not pad_ok:
            return False, f"Buffer padding overwritten by the GPU kernel: {label} {pad_msg}", None

    # 4. additional defense: check if input Buffers were mutated by the GPU kernel
    for i, (input_bytes, snapshot) in enumerate(input_snapshots):
        if not np.array_equal(input_bytes, snapshot):
            return False, f"FAIL: input[{i}] was mutated by the GPU kernel", None

    rtol = getattr(mod, "RTOL", 1e-3)
    atol = getattr(mod, "ATOL", 1e-3)

    diag_msgs = []
    for i, (out_arr, ref_arr) in enumerate(zip(aligned_outputs, refs, strict=True)):
        ok, msg = verify_output(out_arr, np.asarray(ref_arr), rtol=rtol, atol=atol)
        if not ok:
            tag = f"out[{i}] " if not single_output else ""
            return False, f"FAIL ({tag}{msg})", None
        diag_msgs.append(msg)

    perf_str = ""
    if hasattr(mod, "metrics"):
        perf_str = f" | {mod.metrics(case, res.median_ms)}"

    summary = (
        f"PASS ({'; '.join(diag_msgs)}) | "
        f"median={res.median_ms:.4f} ms, min={res.min_ms:.4f} ms{perf_str}"
    )
    return True, summary, res.median_ms


def main():
    parser = argparse.ArgumentParser(description="Local Metal LeetGPU Runner")
    parser.add_argument(
        "name", help="Problem name (loads problems/<name>.py & shaders/starters/<name>.metal)"
    )
    parser.add_argument("--shader", type=Path, default=None, help="Override path to .metal file")
    parser.add_argument(
        "--entry", type=str, default=None, help="Override kernel entry function name"
    )
    parser.add_argument(
        "--lib",
        type=Path,
        default=Path("build/libleetgpu_metal.dylib"),
        help="Path to harness dylib",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup dispatches")
    parser.add_argument("--runs", type=int, default=15, help="Number of timed dispatches")
    parser.add_argument(
        "--no-flush",
        action="store_true",
        help="Disable 64MB L2/SLC cache flush before timing",
    )
    parser.add_argument(
        "--safe-math",
        action="store_true",
        help="Compile MSL with MTLMathModeSafe (disable fast-math)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for input generation")
    args = parser.parse_args()
    if error := run_count_error(args.warmup, args.runs):
        parser.error(error)

    problem_path = Path("problems") / f"{args.name}.py"
    shader_path = args.shader if args.shader else Path("shaders/starters") / f"{args.name}.metal"

    if not shader_path.exists():
        print(f"[ERROR] Shader file not found: {shader_path}", file=sys.stderr)
        sys.exit(1)

    lib = load_harness_lib(args.lib)
    mod = load_problem_module(problem_path)
    msl_bytes = shader_path.read_bytes()
    entry_name = (args.entry or getattr(mod, "ENTRY", "solve")).encode("utf-8")

    rng = np.random.default_rng(args.seed)
    cases = mod.TEST_CASES

    print(f"=== Benchmarking: {args.name} ({shader_path}) ===")
    print(
        f"Config: warmup={args.warmup}, runs={args.runs}, "
        f"flush_l2={not args.no_flush}, fast_math={not args.safe_math}\n"
    )

    all_passed = True
    for idx, case in enumerate(cases):
        case_label = ", ".join(f"{k}={v}" for k, v in case.items())
        ok, msg, _ = run_single_case(
            lib=lib,
            mod=mod,
            msl_bytes=msl_bytes,
            entry_name=entry_name,
            case=case,
            rng=rng,
            warmup=args.warmup,
            runs=args.runs,
            flush_l2=not args.no_flush,
            fast_math=not args.safe_math,
        )
        print(f"[Case {idx + 1}/{len(cases)}] ({case_label})")
        print(f"  -> {msg}")
        if not ok:
            all_passed = False
            break

    print("\n" + ("ALL CASES PASSED." if all_passed else "BENCHMARK FAILED."))
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
