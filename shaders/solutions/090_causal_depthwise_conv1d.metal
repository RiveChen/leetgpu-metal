// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& taps [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  uint channel = i % width, time = (i / width) % length;
  float sum = bias[channel];
  for (uint lag = 0; lag < taps && lag <= time; ++lag)
    sum += x[i - lag * width] * weight[channel * taps + lag];
  out[i] = sum;
}
