// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant uint& sinks [[buffer(6)]],
    constant uint& window [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  uint begin = 0;
  uint row = i / width, col = i % width;
  float m = -INFINITY, total = 0, numerator = 0;
  for (uint j = begin; j <= row; ++j) {
    if (j >= sinks && j + window <= row)
      continue;
    float score = 0;
    for (uint d = 0; d < width; ++d)
      score += q[row * width + d] * k[j * width + d];
    score *= rsqrt(float(width));
    float next = max(m, score), rescale = exp(m - next), w = exp(score - next);
    numerator = numerator * rescale + w * v[j * width + col];
    total = total * rescale + w;
    m = next;
  }
  out[i] = numerator / total;
}
