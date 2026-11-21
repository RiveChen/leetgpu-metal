#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* grad [[buffer(0)]],
    device const float* buf [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant float& momentum [[buffer(5)]],
    device float* scratch [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Normalize the momentum-adjusted gradient and run five quintic Newton-Schulz iterations.
}
