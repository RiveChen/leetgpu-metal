#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* volume [[buffer(0)]],
    device const float* kernel_weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& depth [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    constant uint& kd [[buffer(6)]],
    constant uint& kh [[buffer(7)]],
    constant uint& kw [[buffer(8)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute valid 3D cross-correlation in depth, row, column order.
}
