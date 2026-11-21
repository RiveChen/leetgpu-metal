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
    constant uint& window [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {

  uint row = i / width, col = i % width;
  float hi = -INFINITY, mass = 0, sum = 0;
  for (uint j = row - min(row, window); j < min(length, row + window + 1); ++j) {
    float score = 0;
    for (uint d = 0; d < width; ++d)
      score += q[row * width + d] * k[j * width + d];
    score *= rsqrt(float(width));
    float m = max(hi, score), old = exp(hi - m), w = exp(score - m);
    sum = sum * old + w * v[j * width + col];
    mass = mass * old + w;
    hi = m;
  }
  out[i] = sum / mass;
}
