// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= n - k + 1)
    return;
  float sum = 0;
  for (uint j = 0; j < k; ++j)
    sum += x[i + j] * weights[j];
  y[i] = sum;
}
