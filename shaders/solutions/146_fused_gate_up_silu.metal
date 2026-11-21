// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= rows * width)
    return;
  uint row = index / width, col = index % width;
  float gate = 0, up = 0;
  for (uint j = 0; j < hidden; ++j) {
    float v = x[row * hidden + j];
    uint p = j * 2 * width + col;
    gate += v * weights[p];
    up += v * weights[p + width];
  }
  out[index] = (gate / (1 + exp(-gate))) * up;
}
