import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import bench

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("LEETGPU_TEST_LIB", ROOT / "build/libleetgpu_metal.dylib"))
COPY_PREFIX = b"""
#include <metal_stdlib>
using namespace metal;
kernel void solve(device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
                  constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < count) dst[i] = src[i];
}
"""


class RunnerTests(unittest.TestCase):
    def test_page_padding_at_allocation_boundaries(self):
        page = bench.PAGE_SIZE
        for size in (0, page - 1, page, page + 1, 2 * page):
            with self.subTest(size=size):
                arr, mm, payload = bench.alloc_page_aligned((size,), np.uint8)
                self.assertEqual(payload, arr.nbytes)
                self.assertEqual(arr.ctypes.data % page, 0)
                self.assertEqual(len(mm) % page, 0)
                self.assertGreater(len(mm), payload)
                self.assertTrue(bench.check_page_padding(mm, payload)[0])
                mm[payload] = 0
                ok, message = bench.check_page_padding(mm, payload)
                self.assertFalse(ok)
                self.assertIn("byte offset +0", message)

    def test_empty_float_output_requires_matching_shape_and_dtype(self):
        out = np.empty((2, 0), dtype=np.float32)
        for ref, expected in ((out.copy(), True), (out.T, False), (out.astype(np.float64), False)):
            with self.subTest(shape=ref.shape, dtype=ref.dtype):
                ok, message = bench.verify_output(out, ref, 0, 0)
                self.assertEqual(ok, expected, message)

    def test_nan_input_snapshot(self):
        lib = bench.load_harness_lib(LIBRARY)
        # Both bit patterns encode NaN. Only the second case changes the input bytes.
        values = np.array(0x7FC00001, dtype=np.uint32).view(np.float32)
        shader = b"""
        #include <metal_stdlib>
        using namespace metal;
        kernel void solve(device uint *input [[buffer(0)]], constant uint &bits [[buffer(2)]]) {
            if (bits != 0) input[0] = bits;
        }
        """
        problem = SimpleNamespace(
            make_inputs=lambda bits, rng: ([values], (0,), [np.uint32(bits)]),
            reference_impl=lambda src: np.empty(0, dtype=np.float32),
            dispatch_config=lambda case: (1, 1, 1, 1, 1, 1),
        )
        for bits in (0, 0x7FC00002):
            with self.subTest(bits=bits):
                ok, message, _ = bench.run_single_case(
                    lib,
                    problem,
                    shader,
                    b"solve",
                    bits,
                    np.random.default_rng(42),
                    0,
                    1,
                    False,
                    False,
                )
                self.assertEqual(ok, bits == 0, message)
                if bits:
                    self.assertIn("input[0] was mutated", message)

    def test_missing_and_complete_output_writes(self):
        lib = bench.load_harness_lib(LIBRARY)
        for values in (
            np.array([255, 255], dtype=np.uint8),
            np.array([0, -128], dtype=np.int8),
            np.array([False, True], dtype=np.bool_),
        ):
            problem = SimpleNamespace(
                DTYPE=values.dtype,
                make_inputs=lambda count, rng, v=values: ([v], v.shape, [np.uint32(count)]),
                reference_impl=lambda src: src,
                dispatch_config=lambda case: (1, 1, 1, 2, 1, 1),
            )
            for written in (0, 1, 2):
                with self.subTest(dtype=values.dtype, written=written):
                    ok, msg, _ = bench.run_single_case(
                        lib,
                        problem,
                        COPY_PREFIX,
                        b"solve",
                        written,
                        np.random.default_rng(42),
                        0,
                        1,
                        False,
                        False,
                    )
                    self.assertEqual(ok, written == len(values), msg)
                    if not ok:
                        self.assertIn("Exact mismatch", msg)

    def test_native_invalid_counts(self):
        lib = bench.load_harness_lib(LIBRARY)
        for warmup, runs in ((-1, 1), (0, 0), (0, -1)):
            with self.subTest(warmup=warmup, runs=runs):
                result = lib.run_metal_kernel(
                    b"", b"", None, 0, None, 0, bench.DispatchConfig(), warmup, runs, False, False
                )
                self.assertEqual(result.status, -2)
                self.assertIn(b"warmup_runs", result.error_msg)
                self.assertIn(b"timed_runs", result.error_msg)

    def test_cli_invalid_counts(self):
        overflow = str(np.iinfo(np.intc).max + 1)
        for flag, value in (
            ("--runs", "0"),
            ("--runs", "-1"),
            ("--warmup", "-1"),
            ("--runs", overflow),
            ("--warmup", overflow),
        ):
            with self.subTest(flag=flag, value=value):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts/bench.py"), "unused", flag, value],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(f"error: {flag}", result.stderr)
