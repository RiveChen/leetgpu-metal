#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float2* x [[buffer(0)]],
    device float2* y [[buffer(1)]],
    constant uint& h [[buffer(2)]],
    constant uint& w [[buffer(3)]],
    device float2* row_spectra [[buffer(4)]],
    uint kx [[thread_position_in_grid]]
) {
  // row_spectra holds h*w complex values. Each thread owns frequency column kx.
  // TODO: Compute the unnormalized forward 2D Fourier transform of interleaved real/imaginary values.
}
