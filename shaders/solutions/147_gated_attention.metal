// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* x [[buffer(3)]],
    device const float* gate [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& model [[buffer(7)]],
    constant uint& heads [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= length * model)
    return;
  uint row = index / model, col = index % model, width = model / heads, head = col / width;
  uint channel = col % width, qbase = row * model + head * width, first = 0, last = row;

  float peak = -INFINITY, denom = 0, weighted = 0;
  for (uint t = first; t <= last; ++t) {
    uint kbase = t * model + head * width;

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

  float logit = 0;
  for (uint j = 0; j < model; ++j)
    logit += x[row * model + j] * gate[j * model + col];
  out[index] = (weighted / denom) / (1 + exp(-logit));
}
