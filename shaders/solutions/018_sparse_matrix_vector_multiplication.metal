// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& nnz [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows)
    return;
  float sum = 0;
  for (uint j = 0; j < cols; ++j)
    sum += a[i * cols + j] * x[j];
  y[i] = sum;
}
