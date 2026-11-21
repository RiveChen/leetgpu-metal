#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute scaled softmax attention for each row using keys and values up to and including that row.
}
