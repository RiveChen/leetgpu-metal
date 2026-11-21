// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& threshold [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  uint base = i * vocab;
  float m = logits[base], denom = 0;
  for (uint j = 1; j < vocab; ++j)
    m = max(m, logits[base + j]);
  for (uint j = 0; j < vocab; ++j) {
    float w = exp(logits[base + j] - m);
    out[base + j] = w >= threshold ? w : 0;
    denom += out[base + j];
  }
  for (uint j = 0; j < vocab; ++j)
    out[base + j] /= denom;
}
