// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& kv_heads [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= heads * length * width)
    return;
  uint col = i % width, head = i / (length * width), kv = head / (heads / kv_heads);
  device const float* query = q + (i / width) * width;
  float peak = -INFINITY, numerator = 0, denominator = 0;
  for (uint t = 0; t < length; ++t) {
    uint offset = (kv * length + t) * width;
    device const float* keys = k + offset;
    device const float* values = v + offset;
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
