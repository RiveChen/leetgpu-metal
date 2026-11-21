#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    constant int& target [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Count occurrences of target in the entire input array.
}
