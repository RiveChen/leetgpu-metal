#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& features [[buffer(5)]],
    constant uint& outputs [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Evaluate a batched linear layer using output-major weights and one bias per output.
}
