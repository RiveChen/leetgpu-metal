#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& groups [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& inner [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Multiply each row range of A by the B matrix assigned by its group offsets.
}
