#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const float* initial_x [[buffer(2)]],
    device const float* initial_y [[buffer(3)]],
    device int* labels [[buffer(4)]],
    device float* center_x [[buffer(5)]],
    device float* center_y [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    constant uint& steps [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Run steps of Lloyd assignment/update with one 64-thread group. Keep empty centers fixed and break distance ties by the lowest center index.
}
