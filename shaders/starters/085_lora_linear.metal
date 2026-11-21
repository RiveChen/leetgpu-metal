#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& inputs [[buffer(6)]],
    constant uint& outputs [[buffer(7)]],
    constant uint& rank [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply the base linear layer and add the scaled low-rank adapter.
}
