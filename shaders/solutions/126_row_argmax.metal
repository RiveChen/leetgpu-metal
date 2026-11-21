// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& vocab [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch)
    return;
  uint best = 0;
  for (uint j = 1; j < vocab; ++j)
    if (logits[i * vocab + j] > logits[i * vocab + best])
      best = j;
  output[i] = int(best);
}
