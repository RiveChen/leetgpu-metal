// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* advantages [[buffer(0)]],
    device const float* logp [[buffer(1)]],
    device const float* old [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant float& clip [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& steps [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  float sum = 0;
  for (uint j = 0; j < batch * steps; ++j) {
    float ratio = exp(logp[j] - old[j]), a = advantages[j];
    sum += min(ratio * a, clamp(ratio, 1.0f - clip, 1.0f + clip) * a);
  }
  out[0] = -sum / float(batch * steps);
}
