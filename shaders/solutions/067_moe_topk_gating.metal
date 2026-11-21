// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device float* weights [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& experts [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows)
    return;
  device const float* scores = logits + i * experts;
  uint prior = 0;
  float maximum = 0, sum = 0;
  for (uint r = 0; r < k; ++r) {
    uint picked = 0;
    float best = -INFINITY;
    for (uint j = 0; j < experts; ++j) {
      if (r && (scores[j] > scores[prior] || (scores[j] == scores[prior] && j <= prior)))
        continue;
      if (scores[j] > best) {
        best = scores[j];
        picked = j;
      }
    }
    if (r == 0)
      maximum = best;
    float w = exp(best - maximum);
    weights[i * k + r] = w;
    indices[i * k + r] = int(picked);
    sum += w;
    prior = picked;
  }
  for (uint r = 0; r < k; ++r)
    weights[i * k + r] /= sum;
}
