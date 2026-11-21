// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* volume [[buffer(0)]],
    device const float* kernel_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& depth [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& kd [[buffer(6)]],
    constant uint& kh [[buffer(7)]],
    constant uint& kw [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  uint ow = cols - kw + 1, oh = rows - kh + 1;
  if (i >= (depth - kd + 1) * oh * ow)
    return;
  uint x = i % ow, y = (i / ow) % oh, z = i / (ow * oh);
  float sum = 0;
  for (uint dz = 0; dz < kd; ++dz)
    for (uint dy = 0; dy < kh; ++dy)
      for (uint dx = 0; dx < kw; ++dx)
        sum += volume[((z + dz) * rows + y + dy) * cols + x + dx] *
               kernel_weights[(dz * kh + dy) * kw + dx];
  output[i] = sum;
}
