#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const int* x [[buffer(0)]],
    device int* output [[buffer(1)]],
    constant uint& depth [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant int& target [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Count occurrences of target in the entire input array.
}
