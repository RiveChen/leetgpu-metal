// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  uint count = 0;
  for (uint j = 0; j < n; ++j)
    if (x[j] > 0)
      y[count++] = x[j];
  for (uint j = count; j < n; ++j)
    y[j] = 0;
}
