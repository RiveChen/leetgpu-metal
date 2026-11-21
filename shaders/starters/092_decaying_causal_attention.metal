#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& decay [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Accumulate causal scaled dot products with geometric time decay and no softmax.
}
