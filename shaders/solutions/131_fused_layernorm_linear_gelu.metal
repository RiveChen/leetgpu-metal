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
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device const float* weights [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& features [[buffer(7)]],
    constant uint& outputs [[buffer(8)]],
    constant float& eps [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows * outputs)
    return;
  uint row = i / outputs, col = i % outputs;
  device const float* values = x + row * features;
  float mean = 0;
  for (uint k = 0; k < features; ++k)
    mean += values[k];
  mean /= float(features);
  float variance = 0;
  for (uint k = 0; k < features; ++k) {
    float delta = values[k] - mean;
    variance += delta * delta;
  }
  float inv = rsqrt(variance / float(features) + eps);
  float sum = bias[col];
  for (uint k = 0; k < features; ++k)
    sum += ((values[k] - mean) * inv * gamma[k] + beta[k]) * weights[k * outputs + col];
  y[i] = gelu(sum);
}
