#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& h [[buffer(2)]],
    constant uint& w [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Compute the inclusive 2D prefix sum for every pixel.
}
