// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& block [[buffer(7)]],
    constant uint& selected [[buffer(8)]],
    device float* scratch [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  uint h = i / 64, lane = i % 64, blocks = (length + block - 1) / block;
  device float* scores = scratch + h * 2 * blocks;
  device float* included = scores + blocks;
  for (uint b = lane; b < blocks; b += 64) {
    uint begin = b * block, end = min(begin + block, length);
    float score = 0;
    for (uint d = 0; d < width; ++d) {
      float pooled = 0;
      for (uint s = begin; s < end; ++s)
        pooled += k[(h * length + s) * width + d];
      score += q[h * width + d] * (pooled / float(end - begin));
    }
    scores[b] = score;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint b = lane; b < blocks; b += 64) {
    uint rank = 0;
    for (uint j = 0; j < blocks && rank < selected; ++j)
      rank += (scores[j] > scores[b] || (scores[j] == scores[b] && j < b));
    included[b] = rank < selected;
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint d = lane; d < width; d += 64) {
    float peak = -INFINITY, den = 0, acc = 0;
    for (uint s = 0; s < length; ++s) {
      if (!included[s / block])
        continue;
      uint base = (h * length + s) * width;
      float dot = 0;
      for (uint j = 0; j < width; ++j)
        dot += q[h * width + j] * k[base + j];
      float score = dot * rsqrt(float(width)), next = max(peak, score), old = exp(peak - next),
            w = exp(score - next);
      acc = acc * old + w * v[base + d];
      den = den * old + w;
      peak = next;
    }
    output[h * width + d] = acc / den;
  }
}
