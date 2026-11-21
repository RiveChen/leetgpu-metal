#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& depth [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& first_z [[buffer(5)]],
    constant uint& last_z [[buffer(6)]],
    constant uint& first_row [[buffer(7)]],
    constant uint& last_row [[buffer(8)]],
    constant uint& first_col [[buffer(9)]],
    constant uint& last_col [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Sum all elements within the given inclusive index bounds.
}
