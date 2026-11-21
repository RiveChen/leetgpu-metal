#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* image [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    constant uint& kh [[buffer(5)]],
    constant uint& kw [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Blur with the supplied odd-sized Gaussian filter and zero padding.
}
