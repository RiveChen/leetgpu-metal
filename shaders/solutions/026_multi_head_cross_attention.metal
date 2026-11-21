// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& head_count [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= m * head_count)
    return;
  uint head = i % head_count, length = n;
  uint base = i * width;
  for (uint col = 0; col < width; ++col)
    output[base + col] = 0;
  float peak = -INFINITY, denominator = 0;
  for (uint token = 0; token < length; ++token) {
    uint key_base = (token * head_count + head) * width;
    float score = 0;
    for (uint col = 0; col < width; ++col)
      score += q[base + col] * k[key_base + col];
    score *= rsqrt(float(width));
    float next_peak = max(peak, score);
    float rescale = exp(peak - next_peak), weight = exp(score - next_peak);
    for (uint col = 0; col < width; ++col)
      output[base + col] = output[base + col] * rescale + weight * v[key_base + col];
    denominator = denominator * rescale + weight;
    peak = next_peak;
  }
  for (uint col = 0; col < width; ++col)
    output[base + col] /= denominator;
}
