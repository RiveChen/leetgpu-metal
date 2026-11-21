// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* up [[buffer(2)]],
    device const float* down [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows * width)
    return;
  uint row = i / width, col = i % width;
  float sum = 0;
  for (uint j = 0; j < hidden; ++j) {
    float g = 0, u = 0;
    for (uint d = 0; d < width; ++d) {
      float value = x[row * width + d];
      g += value * gate[d * hidden + j];
      u += value * up[d * hidden + j];
    }
    sum += g / (1 + exp(-g)) * u * down[j * width + col];
  }
  y[i] = sum;
}
