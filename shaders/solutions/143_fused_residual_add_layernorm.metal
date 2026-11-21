// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* y [[buffer(4)]],
    device float* combined [[buffer(5)]],
    device float* means [[buffer(6)]],
    device float* inv_std [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= rows)
    return;
  uint base = index * width;
  // Accumulate differences from the first element to retain precision at large offsets.
  float origin = x[base] + residual[base], sum = 0;
  for (uint c = 0; c < width; ++c) {
    float h = x[base + c] + residual[base + c];
    combined[base + c] = h;
    sum += h - origin;
  }
  float offset = sum / width, sq = 0;
  for (uint c = 0; c < width; ++c) {
    float dev = (combined[base + c] - origin) - offset;
    sq += dev * dev;
  }
  float inv = rsqrt(sq / width + eps);
  means[index] = origin + offset;
  inv_std[index] = inv;
  for (uint c = 0; c < width; ++c)
    y[base + c] = ((combined[base + c] - origin) - offset) * inv * gamma[c] + beta[c];
}
