#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* res [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Add the residual, then apply row-wise RMS normalization and per-feature weights.
}
