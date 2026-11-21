// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* cos_table [[buffer(3)]],
    device const float* sin_table [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& kv [[buffer(7)]],
    constant uint& length [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= heads * length * width)
    return;
  uint channel = index % width, row = (index / width) % length, head = index / (width * length);
  uint qbase = index - channel, kvhead = head / (heads / kv), last = row;
  uint first = 0;

  float peak = -INFINITY, denom = 0, weighted = 0;
  for (uint t = first; t <= last; ++t) {
    uint kbase = (kvhead * length + t) * width;

    uint pairs = width / 2;
    float score = 0;
    for (uint j = 0; j < pairs; ++j) {
      float2 query = float2(q[qbase + j], q[qbase + pairs + j]);
      float2 key = float2(k[kbase + j], k[kbase + pairs + j]);
      float cq = cos_table[row * pairs + j], sq = sin_table[row * pairs + j];
      float ck = cos_table[t * pairs + j], sk = sin_table[t * pairs + j];
      query = float2(cq * query.x - sq * query.y, sq * query.x + cq * query.y);
      key = float2(ck * key.x - sk * key.y, sk * key.x + ck * key.y);
      score += dot(query, key);
    }
    score *= rsqrt(float(width));

    float next = max(peak, score);
    float carry = exp(peak - next), weight = exp(score - next);
    weighted = weighted * carry + v[kbase + channel] * weight;
    denom = denom * carry + weight;
    peak = next;
  }
  out[index] = weighted / denom;
}
