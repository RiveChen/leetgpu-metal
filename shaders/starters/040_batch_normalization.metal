#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& samples [[buffer(4)]],
    constant uint& channels [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Normalize each channel over the batch using population variance, then apply gamma and beta.
}
