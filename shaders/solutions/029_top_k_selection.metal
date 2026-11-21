// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* values [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= n)
    return;
  float value = values[i];
  uint rank = 0;
  for (uint j = 0; j < n; ++j) {
    if (values[j] > value || (values[j] == value && j < i))
      ++rank;
    if (rank >= k)
      return;
  }
  output[rank] = value;
}
