// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& queries [[buffer(4)]],
    constant uint& keys [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= queries * width)
    return;
  uint row = i / width, col = i % width;
  float maximum = -INFINITY, denominator = 0, weighted = 0;
  for (uint j = 0; j < keys; ++j) {
    float score = 0;
    for (uint d = 0; d < width; ++d)
      score += q[row * width + d] * k[j * width + d];
    score *= rsqrt(float(width));
    float next_max = max(maximum, score);
    float rescale = exp(maximum - next_max), weight = exp(score - next_max);
    denominator = denominator * rescale + weight;
    weighted = weighted * rescale + weight * v[j * width + col];
    maximum = next_max;
  }
  output[i] = weighted / denominator;
}
