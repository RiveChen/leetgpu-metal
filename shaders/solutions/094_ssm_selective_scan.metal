// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* u [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device const float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device const float* c [[buffer(4)]],
    device const float* skip [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& length [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    constant uint& states [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  uint sample = i / width, channel = i % width;
  float state[64] = {};
  for (uint time = 0; time < length; ++time) {
    uint index = (sample * length + time) * width + channel;
    uint base = (sample * length + time) * states;
    float dt = delta[index], input = u[index], sum = skip[channel] * input;
    for (uint s = 0; s < states; ++s) {
      state[s] = exp(dt * a[channel * states + s]) * state[s] + dt * b[base + s] * input;
      sum += state[s] * c[base + s];
    }
    out[index] = sum;
  }
}
