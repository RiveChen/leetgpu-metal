#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& first_i [[buffer(3)]],
    constant uint& last_i [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Sum all elements within the given inclusive index bounds.
}
