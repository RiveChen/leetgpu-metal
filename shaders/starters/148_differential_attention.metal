#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* lambda_head [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& length [[buffer(5)]],
    constant uint& heads [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    constant float& initial [[buffer(8)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Subtract the two causal attention maps and RMS-normalize each full head output.
}
