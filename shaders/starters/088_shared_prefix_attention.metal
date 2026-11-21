#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* shared_k [[buffer(1)]],
    device const float* shared_v [[buffer(2)]],
    device const float* local_k [[buffer(3)]],
    device const float* local_v [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& prefix [[buffer(8)]],
    constant uint& suffix [[buffer(9)]],
    constant uint& width [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Attend to the full shared prefix and the causal portion of the per-sequence suffix.
}
