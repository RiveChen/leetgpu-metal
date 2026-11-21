// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* dt [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* decay [[buffer(3)]],
    device const float* b [[buffer(4)]],
    device const float* c [[buffer(5)]],
    device const float* skip [[buffer(6)]],
    device const float* z [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant uint& batch [[buffer(9)]],
    constant uint& length [[buffer(10)]],
    constant uint& heads [[buffer(11)]],
    constant uint& width [[buffer(12)]],
    constant uint& groups [[buffer(13)]],
    constant uint& states [[buffer(14)]],
    device float* state [[buffer(15)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= batch * heads * width)
    return;
  uint channel = index % width, head = (index / width) % heads, sample = index / (heads * width);
  device float* memory = state + index * states;
  for (uint n = 0; n < states; ++n)
    memory[n] = 0;
  for (uint t = 0; t < length; ++t) {
    uint pos = (sample * length + t) * heads + head, elem = pos * width + channel;
    uint base = ((sample * length + t) * groups + head / (heads / groups)) * states;
    float raw = dt[pos] + bias[head];
    float tail = exp(-abs(raw));
    float delta = max(raw, 0.f) + (tail < 1e-4f ? tail : log(1.f + tail));
    float factor = exp(delta * decay[head]), injected = delta * x[elem], value = 0;
    for (uint n = 0; n < states; ++n) {
      memory[n] = factor * memory[n] + injected * b[base + n];
      value += memory[n] * c[base + n];
    }
    out[elem] = (value + skip[head] * x[elem]) * (z[elem] / (1 + exp(-z[elem])));
  }
}
