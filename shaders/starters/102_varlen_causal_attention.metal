#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const int* offsets [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& width [[buffer(6)]],
    constant uint& sequences [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute causal self-attention independently in each packed sequence.
}
