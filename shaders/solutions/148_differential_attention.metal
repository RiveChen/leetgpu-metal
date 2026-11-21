// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float2 scaled_scores(device const float* q, device const float* k, uint width) {
  float2 pair = 0;
  for (uint j = 0; j < width; ++j)
    pair += float2(q[j] * k[j], q[width + j] * k[width + j]);
  return pair * rsqrt(float(width));
}

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* lambda_head [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant float& initial [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= length * heads)
    return;
  uint head = index % heads, row = index / heads, span = 2 * width, base = index * span;
  float2 peak = float2(-INFINITY), denom = 0;
  for (uint t = 0; t <= row; ++t) {
    float2 score = scaled_scores(q + base, k + (t * heads + head) * span, width);
    float2 next = max(peak, score);
    denom = denom * exp(peak - next) + exp(score - next);
    peak = next;
  }
  for (uint j = 0; j < span; ++j)
    out[base + j] = 0;
  for (uint t = 0; t <= row; ++t) {
    uint start = (t * heads + head) * span;
    float2 weight = exp(scaled_scores(q + base, k + start, width) - peak) / denom;
    float delta = weight.x - lambda_head[head] * weight.y;
    for (uint j = 0; j < span; ++j)
      out[base + j] += delta * v[start + j];
  }
  float sq = 0;
  for (uint j = 0; j < span; ++j) {
    float val = out[base + j];
    sq += val * val;
  }
  float factor = (1 - initial) * rsqrt(sq / span + 1e-5f);
  for (uint j = 0; j < span; ++j)
    out[base + j] *= factor;
}
