#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant uint& height [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& groups [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Normalize each NCHW group using population variance, then apply per-channel affine parameters.
}
