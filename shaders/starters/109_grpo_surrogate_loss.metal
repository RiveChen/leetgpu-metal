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
  // TODO: Compute the mean GRPO loss using group-normalized rewards, ratio clipping and KL penalty.
}
