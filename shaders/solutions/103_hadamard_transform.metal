// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& width [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  uint base = (i / width) * width, row = i % width;
  float sum = 0;
  for (uint j = 0; j < width; ++j)
    sum += (popcount(row & j) & 1u) ? -x[base + j] : x[base + j];
  out[i] = sum * rsqrt(float(width));
}
