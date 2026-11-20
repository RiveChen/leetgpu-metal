#include <metal_stdlib>
using namespace metal;

// Problem 021: ReLU (Rectified Linear Unit)
//
// Starter Framework:
// Compute y_i = max(0.0, x_i) for each element in the array of length N.
//
// Optimization Tips:
// 1. Vectorized memory transactions: Cast pointers to `device const float4*` and `device float4*`
//    to read/write 128 bits per thread instruction, maximizing memory bus utilization.
// 2. Loop unrolling: Process multiple elements per thread to amortize loop and dispatch overhead.

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
  if (gid < N) {
    output[gid] = max(0.0f, input[gid]);
  }
}
