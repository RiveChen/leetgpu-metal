// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const int* routes [[buffer(1)]],
    device float* dispatched [[buffer(2)]],
    device int* counts [[buffer(3)]],
    constant uint& tokens [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& experts [[buffer(6)]],
    constant uint& capacity [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  uint count = 0;
  uint base = i * capacity * width;
  for (uint token = 0; token < tokens; ++token) {
    if (uint(routes[token]) != i)
      continue;
    for (uint d = 0; d < width; ++d)
      dispatched[base + count * width + d] = x[token * width + d];
    ++count;
  }
  counts[i] = count;
  for (uint j = count * width; j < capacity * width; ++j)
    dispatched[base + j] = 0;
}
