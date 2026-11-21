// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* cache_k [[buffer(1)]],
    device const float* cache_v [[buffer(2)]],
    device const float* tree_k [[buffer(3)]],
    device const float* tree_v [[buffer(4)]],
    device const int* parents [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& prefix [[buffer(7)]],
    constant uint& nodes [[buffer(8)]],
    constant uint& heads [[buffer(9)]],
    constant uint& width [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= nodes * heads * width)
    return;
  uint d = i % width, h = (i / width) % heads, t = i / (heads * width);
  float peak = -INFINITY, den = 0, sum = 0;
  int node = int(t);
  // Visit the full prefix, then only this node and its ancestors.
  for (uint s = 0; s < prefix || node >= 0; ++s) {
    bool cached = s < prefix;
    uint index = cached ? s : uint(node);
    device const float* key = (cached ? cache_k : tree_k) + (index * heads + h) * width;
    device const float* value = (cached ? cache_v : tree_v) + (index * heads + h) * width;
    float score = 0;
    for (uint j = 0; j < width; ++j)
      score += q[i - d + j] * key[j];
    score *= rsqrt(float(width));
    float next = max(peak, score), old = exp(peak - next), w = exp(score - next);
    sum = sum * old + w * value[d];
    den = den * old + w;
    peak = next;
    if (!cached)
      node = parents[node];
  }
  output[i] = sum / den;
}
