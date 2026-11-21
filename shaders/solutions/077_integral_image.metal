// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& h [[buffer(2)]],
    constant uint& w [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= h)
    return;
  float area = 0;
  for (uint col = 0; col < w; ++col) {
    for (uint row = 0; row <= i; ++row)
      area += x[row * w + col];
    y[i * w + col] = area;
  }
}
