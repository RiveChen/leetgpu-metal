#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* values [[buffer(0)]],
    device const int2* shapes [[buffer(1)]],
    device const float2* locations [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& queries [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& levels [[buffer(8)]],
    constant uint& points [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Sum bilinear samples of each level and point using zero padding.
}
