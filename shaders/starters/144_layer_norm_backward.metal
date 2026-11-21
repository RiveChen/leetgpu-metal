#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* mean [[buffer(2)]],
    device const float* rstd [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* dx [[buffer(5)]],
    device float* dgamma [[buffer(6)]],
    device float* dbeta [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& channels [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Use saved row statistics to compute layer normalization input, scale and bias gradients.
}
