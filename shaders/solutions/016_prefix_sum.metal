// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

// Problem 016: Prefix Sum (Inclusive Scan)
//
// Task:
// Compute output[i] = sum_{j=0}^i input[j] for i in [0, N-1].
//
// Optimization Tips:
// 1. Metal Native Hardware Primitives: Apple Silicon supports `simd_prefix_inclusive_sum(val)`
//    which performs single-instruction warp-level scan with zero shared memory latency.
// 2. Multi-block Scan: For N > 1024, use the classical Blelloch work-efficient parallel scan
//    or decoupled look-back (single-pass prefix scan).

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]
) {
  // Baseline starter: sequential scan across elements for correctness verification
  if (tid == 0) {
    float acc = 0.0f;
    for (uint i = 0; i < N; ++i) {
      acc += input[i];
      output[i] = acc;
    }
  }
}
