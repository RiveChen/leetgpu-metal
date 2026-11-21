#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& block [[buffer(7)]],
    constant uint& selected [[buffer(8)]],
    device float* scratch [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Rank mean-key blocks, then attend over all tokens in the selected blocks.
}
