// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const int* table [[buffer(3)]],
    device const int* contexts [[buffer(4)]],
    device const int* offsets [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& seqs [[buffer(7)]],
    constant uint& total [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& kv [[buffer(10)]],
    constant uint& width [[buffer(11)]],
    constant uint& block [[buffer(12)]],
    constant uint& slots [[buffer(13)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= total * heads * width)
    return;
  uint token = index / (heads * width), head = (index / width) % heads, channel = index % width,
       qbase = index - channel;
  uint seq = 0;
  while (seq + 1 < seqs && token >= uint(offsets[seq + 1]))
    ++seq;
  uint first = 0, last = uint(contexts[seq]) + token - uint(offsets[seq + 1]);

  float peak = -INFINITY, denom = 0, weighted = 0;
  for (uint t = first; t <= last; ++t) {

    uint page = uint(table[seq * slots + t / block]);
    uint kbase = ((page * block + t % block) * kv + head / (heads / kv)) * width;

    float score = 0;
    for (uint j = 0; j < width; ++j)
      score += q[qbase + j] * k[kbase + j];
    score *= rsqrt(float(width));

    float next = max(peak, score);
    float carry = exp(peak - next), weight = exp(score - next);
    weighted = weighted * carry + v[kbase + channel] * weight;
    denom = denom * carry + weight;
    peak = next;
  }
  out[index] = weighted / denom;
}
