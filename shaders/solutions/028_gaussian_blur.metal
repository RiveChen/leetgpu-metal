// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* image [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& kh [[buffer(5)]],
    constant uint& kw [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows * cols)
    return;
  float sum = 0;
  for (uint y = 0; y < kh; ++y) {
    int row = int(i / cols) + int(y) - int(kh / 2);
    if (row < 0 || row >= int(rows))
      continue;
    for (uint x = 0; x < kw; ++x) {
      int col = int(i % cols) + int(x) - int(kw / 2);
      if (col >= 0 && col < int(cols))
        sum += image[row * cols + col] * weights[y * kw + x];
    }
  }
  output[i] = sum;
}
