// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float2 rotate_phase(float2 value, uint phase, uint period) {
  float cosine;
  float sine = sincos(-2 * M_PI_F * float(phase) / float(period), cosine);
  return float2(value.x * cosine - value.y * sine, value.y * cosine + value.x * sine);
}

kernel void solve(
    device const float2* x [[buffer(0)]],
    device float2* y [[buffer(1)]],
    constant uint& h [[buffer(2)]],
    constant uint& w [[buffer(3)]],
    device float2* row_spectra [[buffer(4)]],
    uint kx [[thread_position_in_grid]]
) {
  if (kx >= w)
    return;
  // Each thread owns one frequency column in both row_spectra and the output.
  for (uint row = 0; row < h; ++row) {
    float2 sum = 0;
    for (uint col = 0; col < w; ++col)
      sum += rotate_phase(x[row * w + col], (col * kx) % w, w);
    row_spectra[row * w + kx] = sum;
  }
  for (uint ky = 0; ky < h; ++ky) {
    float2 sum = 0;
    for (uint row = 0; row < h; ++row)
      sum += rotate_phase(row_spectra[row * w + kx], (row * ky) % h, h);
    y[ky * w + kx] = sum;
  }
}
