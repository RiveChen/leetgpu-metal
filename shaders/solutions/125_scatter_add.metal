// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* src [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& groups [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= groups * width)
    return;
  uint group = i / width, col = i % width;
  float sum = 0;
  for (uint j = 0; j < rows; ++j)
    if (uint(indices[j]) == group)
      sum += src[j * width + col];
  output[i] = sum;
}
