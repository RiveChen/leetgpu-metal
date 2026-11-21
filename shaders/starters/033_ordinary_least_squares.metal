#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device float* beta [[buffer(2)]],
    constant uint& samples [[buffer(3)]],
    constant uint& features [[buffer(4)]],
    device float* workspace [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Fit ordinary least squares without an intercept on a full-rank design matrix.
}
