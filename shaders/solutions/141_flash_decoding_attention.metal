// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& queries [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  uint element = i / 64, lane = i % 64, d = element % width, h = (element / width) % heads;
  float peak = -INFINITY, den = 0, total = 0;
  for (uint s = lane; s < length; s += 64) {
    uint base = (s * heads + h) * width;
    float score = 0;
    for (uint j = 0; j < width; ++j)
      score += q[element - d + j] * k[base + j];
    score *= rsqrt(float(width));
    float next = max(score, peak), old = exp(peak - next), w = exp(score - next);
    total = total * old + w * v[base + d];
    den = den * old + w;
    peak = next;
  }
  threadgroup float peaks[64], sums[64], values[64];
  peaks[lane] = peak;
  sums[lane] = den;
  values[lane] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint step = 32; step; step /= 2) {
    if (lane < step && sums[lane + step] > 0) {
      float next = max(peaks[lane], peaks[lane + step]);
      float left = exp(peaks[lane] - next), right = exp(peaks[lane + step] - next);
      sums[lane] = left * sums[lane] + right * sums[lane + step];
      values[lane] = left * values[lane] + right * values[lane + step];
      peaks[lane] = next;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0)
    output[element] = values[0] / sums[0];
}
