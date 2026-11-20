#include <metal_stdlib>
using namespace metal;

// Problem 008: Matrix Addition
//
// Starter Framework:
// Compute C = A + B for two N x N square matrices.
//
// Optimization Tips:
// 1. Vectorized Loads/Stores: Cast pointers to float4 to process 4 elements per instruction.
// 2. 2D Thread Tiling: Each thread can calculate a 2x2 or 4x4 sub-tile to increase compute-to-load ratio.

kernel void solve(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
  uint col = gid.x;
  uint row = gid.y;

  if (row < N && col < N) {
    uint idx = row * N + col;
    C[idx] = A[idx] + B[idx];
  }
}
