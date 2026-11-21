#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* cos_table [[buffer(3)]],
    device const float* sin_table [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& kv [[buffer(7)]],
    constant uint& length [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Apply split-half RoPE to Q and K before grouped causal attention.
}
