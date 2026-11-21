#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* parts [[buffer(0)]],
    device const float* lse [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& splits [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Merge split attention results with stable per-head log-sum-exp weights.
}
