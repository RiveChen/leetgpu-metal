#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* dt [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device const float* decay [[buffer(3)]],
    device const float* b [[buffer(4)]],
    device const float* c [[buffer(5)]],
    device const float* skip [[buffer(6)]],
    device const float* z [[buffer(7)]],
    device float* out [[buffer(8)]],
    constant uint& batch [[buffer(9)]],
    constant uint& length [[buffer(10)]],
    constant uint& heads [[buffer(11)]],
    constant uint& width [[buffer(12)]],
    constant uint& groups [[buffer(13)]],
    constant uint& states [[buffer(14)]],
    device float* state [[buffer(15)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Scan the grouped Mamba-2 state across time, then apply the skip connection and SiLU gate.
}
