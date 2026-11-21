// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* chosen [[buffer(0)]],
    device const float* rejected [[buffer(1)]],
    device const float* ref_chosen [[buffer(2)]],
    device const float* ref_rejected [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant float& beta [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  float total = 0;
  for (uint j = 0; j < batch; ++j) {
    float z = beta * ((rejected[j] - chosen[j]) - (ref_rejected[j] - ref_chosen[j]));
    float tail = exp(-fabs(z));
    total += max(z, 0.0f) + (tail < 1e-4f ? tail : log(1.0f + tail));
  }
  out[0] = total / float(batch);
}
