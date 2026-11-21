// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* alpha [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& length [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch * width)
    return;
  uint b = i / width, col = i % width;
  // The challenge bounds the key dimension at 128.
  float state[128] = {};
  for (uint t = 0; t < length; ++t) {
    uint base = (b * length + t) * width;
    float predicted = 0;
    for (uint j = 0; j < width; ++j) {
      state[j] *= alpha[base + j];
      predicted += k[base + j] * state[j];
    }
    float correction = beta[b * length + t] * (v[base + col] - predicted);
    float reading = 0;
    for (uint j = 0; j < width; ++j) {
      state[j] += k[base + j] * correction;
      reading += q[base + j] * state[j];
    }
    output[base + col] = reading;
  }
}
