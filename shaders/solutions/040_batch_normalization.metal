// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& samples [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= channels)
    return;
  float mean = 0;
  for (uint row = 0; row < samples; ++row)
    mean += x[row * channels + i];
  mean /= float(samples);
  float variance = 0;
  for (uint row = 0; row < samples; ++row) {
    float delta = x[row * channels + i] - mean;
    variance += delta * delta;
  }
  float scale = gamma[i] * rsqrt(variance / float(samples) + eps);
  for (uint row = 0; row < samples; ++row) {
    uint index = row * channels + i;
    output[index] = (x[index] - mean) * scale + beta[i];
  }
}
