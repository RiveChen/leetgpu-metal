#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    device float* y [[buffer(4)]],
    device float* combined [[buffer(5)]],
    device float* means [[buffer(6)]],
    device float* inv_std [[buffer(7)]],
    constant uint& rows [[buffer(8)]],
    constant uint& width [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint index [[thread_position_in_grid]]
) {
  // TODO: Add the residual and save the normalized output, sum, row mean and reciprocal standard deviation.
}
