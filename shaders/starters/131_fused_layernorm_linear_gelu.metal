#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& features [[buffer(7)]],
    constant uint& outputs [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Apply affine layer normalization, project features with bias, and apply erf-based GELU.
}
