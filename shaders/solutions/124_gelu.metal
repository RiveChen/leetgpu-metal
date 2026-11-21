// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float gelu(float x) {
  // Abramowitz and Stegun 7.1.26 evaluates the Gaussian tail via erfc.
  float a = fabs(x) * 0.7071067811865475f;
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float p =
      (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t +
        0.254829592f) *
       t);
  float tail = 0.5f * p * exp(-a * a);
  return x * (x < 0 ? tail : 1.0f - tail);
}

kernel void solve(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  if (i < n)
    y[i] = gelu(x[i]);
}
