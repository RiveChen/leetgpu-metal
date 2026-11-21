#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dx [[buffer(3)]],
    device float* dweight [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& channels [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Differentiate row-wise affine RMSNorm with respect to the input and scale.
}
