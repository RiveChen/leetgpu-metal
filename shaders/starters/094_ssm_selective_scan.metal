#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* u [[buffer(0)]],
    device const float* delta [[buffer(1)]],
    device const float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device const float* c [[buffer(4)]],
    device const float* skip [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& length [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    constant uint& states [[buffer(10)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Scan the selective state-space recurrence over time and apply the skip connection.
}
