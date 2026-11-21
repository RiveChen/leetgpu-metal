#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* grad [[buffer(3)]],
    device float* dq [[buffer(4)]],
    device float* dk [[buffer(5)]],
    device float* dv [[buffer(6)]],
    constant uint& queries [[buffer(7)]],
    constant uint& keys [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Differentiate scaled softmax attention with respect to Q, K and V.
}
