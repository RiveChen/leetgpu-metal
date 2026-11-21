#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* logits [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant float& threshold [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& vocab [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Discard probabilities below min_p times the largest probability, then renormalize.
}
