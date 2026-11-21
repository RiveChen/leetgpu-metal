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
  // TODO: Compute mean DPO loss from policy and reference sequence log-probabilities.
}
