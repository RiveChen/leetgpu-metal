// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* previous [[buffer(0)]],
    device const float* logp [[buffer(1)]],
    device float* scores [[buffer(2)]],
    device int* parents [[buffer(3)]],
    device int* tokens [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& beams [[buffer(6)]],
    constant uint& vocab [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  uint offset = i * beams;
  for (uint rank = 0; rank < beams; ++rank) {
    float best = -INFINITY;
    uint pick = 0;
    for (uint j = 0; j < beams * vocab; ++j) {
      float score = previous[offset + j / vocab] + logp[offset * vocab + j];
      if (rank && score >= scores[offset + rank - 1])
        continue;
      if (score > best) {
        best = score;
        pick = j;
      }
    }
    scores[offset + rank] = best;
    parents[offset + rank] = pick / vocab;
    tokens[offset + rank] = pick % vocab;
  }
}
