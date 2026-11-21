// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const char* keys [[buffer(1)]],
    device const char* values [[buffer(2)]],
    device const float* key_scale [[buffer(3)]],
    device const float* value_scale [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& length [[buffer(7)]],
    constant uint& width [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  uint head = i / width, col = i % width;
  float maximum = -INFINITY, total = 0, result = 0;
  for (uint j = 0; j < length; ++j) {
    uint token = head * length + j;
    float score = 0;
    for (uint d = 0; d < width; ++d)
      score += q[head * width + d] * float(keys[token * width + d]);
    score *= key_scale[token] * rsqrt(float(width));
    float m = max(maximum, score), rescale = exp(maximum - m), w = exp(score - m);
    total = total * rescale + w;
    result = result * rescale + w * value_scale[token] * float(values[token * width + col]);
    maximum = m;
  }
  out[i] = result / total;
}
