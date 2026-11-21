// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* rewards [[buffer(0)]],
    device const float* logp [[buffer(1)]],
    device const float* old [[buffer(2)]],
    device const float* ref_logp [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant float& clip [[buffer(5)]],
    constant float& beta [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& groups [[buffer(8)]],
    constant uint& steps [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  float loss = 0;
  for (uint b = 0; b < batch; ++b) {
    float mean = 0, variance = 0;
    for (uint g = 0; g < groups; ++g)
      mean += rewards[b * groups + g];
    mean /= float(groups);
    for (uint g = 0; g < groups; ++g) {
      float d = rewards[b * groups + g] - mean;
      variance += d * d;
    }
    float stddev = sqrt(variance / float(groups));
    for (uint g = 0; g < groups; ++g) {
      float advantage = (rewards[b * groups + g] - mean) / (stddev + 1e-8f);
      for (uint t = 0; t < steps; ++t) {
        uint index = (b * groups + g) * steps + t;
        float ratio = exp(logp[index] - old[index]), delta = ref_logp[index] - logp[index];
        float kl = fabs(delta) < 0.01f
                       ? delta * delta * (0.5f + delta * (1.0f / 6.0f + delta / 24.0f))
                       : exp(delta) - 1.0f - delta;
        loss +=
            beta * kl - min(ratio * advantage, clamp(ratio, 1.0f - clip, 1.0f + clip) * advantage);
      }
    }
  }
  out[0] = loss / float(batch * groups * steps);
}
