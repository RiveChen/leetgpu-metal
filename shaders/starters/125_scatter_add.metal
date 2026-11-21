#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* src [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& groups [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Accumulate source rows into indexed output rows and zero untouched rows.
}
