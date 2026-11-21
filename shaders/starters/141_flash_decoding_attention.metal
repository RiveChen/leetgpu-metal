#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& queries [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Split the KV cache, accumulate online softmax statistics and combine the splits.
}
