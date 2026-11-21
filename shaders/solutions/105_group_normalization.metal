// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& groups [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  uint per = channels / groups * height * width, base = i * per;
  float mean = 0, var = 0;
  for (uint j = 0; j < per; ++j)
    mean += x[base + j];
  mean /= float(per);
  for (uint j = 0; j < per; ++j) {
    float d = x[base + j] - mean;
    var += d * d;
  }
  float inv = rsqrt(var / float(per) + eps);
  for (uint j = 0; j < per; ++j) {
    uint channel = ((base + j) / (height * width)) % channels;
    out[base + j] = (x[base + j] - mean) * inv * gamma[channel] + beta[channel];
  }
}
