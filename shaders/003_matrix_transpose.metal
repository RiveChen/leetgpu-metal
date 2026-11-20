#include <metal_stdlib>
using namespace metal;

// Problem 003: Matrix Transpose
//
// Starter Framework:
// Input matrix is of shape (rows, cols), stored row-major.
// Output matrix must be of shape (cols, rows), stored row-major.
//
// Optimization Tips:
// 1. Naive implementation suffers from non-coalesced writes or reads.
// 2. Use 2D threadgroup shared memory (tile[16][17]) to coalesce both global reads and writes.
//    Adding +1 padding (e.g. 17) avoids SIMD bank conflicts in threadgroup memory.

kernel void solve(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
  uint c = gid.x;
  uint r = gid.y;

  if (r < rows && c < cols) {
    output[c * rows + r] = input[r * cols + c];
  }
}
