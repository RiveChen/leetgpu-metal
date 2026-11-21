// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const int* table [[buffer(3)]],
    device const int* lengths [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    constant uint& block [[buffer(9)]],
    constant uint& pages [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch * heads * width)
    return;
  uint col = i % width, head = (i / width) % heads, b = i / (heads * width);
  device const float* query = q + i - col;
  float peak = -INFINITY, numerator = 0, denominator = 0;
  for (uint t = 0; t < uint(lengths[b]); ++t) {
    uint page = uint(table[b * pages + t / block]);
    uint offset = ((page * block + t % block) * heads + head) * width;
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
