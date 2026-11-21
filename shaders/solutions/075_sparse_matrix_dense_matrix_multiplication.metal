// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& nnz [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= m * k)
    return;
  uint row = i / k, col = i % k;
  float sum = 0;
  for (uint j = 0; j < n; ++j) {
    float value = a[row * n + j];
    if (value != 0)
      sum += value * b[j * k + col];
  }
  c[i] = sum;
}
