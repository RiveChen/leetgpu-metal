// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* parts [[buffer(0)]],
    device const float* lse [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& splits [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= heads * width)
    return;
  uint head = i / width;
  float largest = lse[head];
  for (uint s = 1; s < splits; ++s)
    largest = max(largest, lse[s * heads + head]);
  float num = 0, den = 0;
  for (uint s = 0; s < splits; ++s) {
    float w = exp(lse[s * heads + head] - largest);
    num += w * parts[s * heads * width + i];
    den += w;
  }
  output[i] = num / den;
}
