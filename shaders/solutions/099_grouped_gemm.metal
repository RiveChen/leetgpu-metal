// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  uint row = i / cols, col = i % cols, group = 0;
  while (group + 1 < groups && row >= uint(offsets[group + 1]))
    ++group;
  float sum = 0;
  for (uint k = 0; k < inner; ++k)
    sum += a[row * inner + k] * b[(group * inner + k) * cols + col];
  out[i] = sum;
}
