// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* shared_k [[buffer(1)]],
    device const float* shared_v [[buffer(2)]],
    device const float* local_k [[buffer(3)]],
    device const float* local_v [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& prefix [[buffer(8)]],
    constant uint& suffix [[buffer(9)]],
    constant uint& width [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch * heads * suffix * width)
    return;
  uint col = i % width, pos = (i / width) % suffix, bh = i / (suffix * width), head = bh % heads;
  device const float* query = q + i - col;
  float peak = -INFINITY, numerator = 0, denominator = 0;
  for (uint t = 0; t < prefix + pos + 1; ++t) {
    uint offset = t < prefix ? (head * prefix + t) * width : (bh * suffix + t - prefix) * width;
    device const float* keys = (t < prefix ? shared_k : local_k) + offset;
    device const float* values = (t < prefix ? shared_v : local_v) + offset;
    float score = 0;
    for (uint d = 0; d < width; ++d)
      score += query[d] * keys[d];
    score *= rsqrt(float(width));
    float next_peak = max(peak, score);
    float old_scale = exp(peak - next_peak), weight = exp(score - next_peak);
    numerator = numerator * old_scale + weight * values[col];
    denominator = denominator * old_scale + weight;
    peak = next_peak;
  }
  output[i] = numerator / denominator;
}
