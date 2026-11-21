// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device uint* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& rounds [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= n)
    return;
  uint word = uint(x[i]);
  for (uint r = 0; r < rounds; ++r) {
    uint hash = 2166136261u;
    for (uint b = 0; b < 4; ++b)
      hash = (hash ^ ((word >> (8 * b)) & 255u)) * 16777619u;
    word = hash;
  }
  y[i] = word;
}
