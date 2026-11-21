#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const char* a [[buffer(0)]],
    device const char* b [[buffer(1)]],
    device char* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& scale_a [[buffer(6)]],
    constant float& scale_b [[buffer(7)]],
    constant float& scale_c [[buffer(8)]],
    constant int& zero_a [[buffer(9)]],
    constant int& zero_b [[buffer(10)]],
    constant int& zero_c [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Multiply asymmetric INT8 matrices, round to nearest even and saturate the requantized output.
}
