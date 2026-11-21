// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& kv [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& window [[buffer(8)]],
    constant float& cap [[buffer(9)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= heads * length * width)
    return;
  uint channel = index % width, row = (index / width) % length, head = index / (width * length);
  uint qbase = index - channel, kvhead = head / (heads / kv), last = row;
  uint first = row + 1 > window ? row + 1 - window : 0;

  float peak = -INFINITY, denom = 0, weighted = 0;
  for (uint t = first; t <= last; ++t) {
    uint kbase = (kvhead * length + t) * width;

    float score = 0;
    for (uint j = 0; j < width; ++j)
      score += q[qbase + j] * k[kbase + j];
    score *= rsqrt(float(width));
    score = cap * tanh(score / cap);

    float next = max(peak, score);
    float carry = exp(peak - next), weight = exp(score - next);
    weighted = weighted * carry + v[kbase + channel] * weight;
    denom = denom * carry + weight;
    peak = next;
  }
  out[index] = weighted / denom;
}
