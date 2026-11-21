#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& kv [[buffer(5)]],
    constant uint& length [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant uint& window [[buffer(8)]],
    constant float& cap [[buffer(9)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Compute grouped causal attention with a sliding window and tanh-softcapped logits.
}
