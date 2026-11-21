// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* res [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows)
    return;
  float squares = 0;
  for (uint j = 0; j < width; ++j) {
    uint pos = i * width + j;
    float z = x[pos] + res[pos];
    squares += z * z;
  }
  float inv = rsqrt(squares / float(width) + eps);
  for (uint j = 0; j < width; ++j) {
    uint pos = i * width + j;
    y[pos] = (x[pos] + res[pos]) * weight[j] * inv;
  }
}
