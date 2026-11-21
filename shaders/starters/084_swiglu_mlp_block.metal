#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* up [[buffer(2)]],
    device const float* down [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& hidden [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply the gate and up projections, multiply SiLU(gate) by up, and project down.
}
