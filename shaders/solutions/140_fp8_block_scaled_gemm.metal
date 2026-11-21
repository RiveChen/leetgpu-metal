// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float decode(uchar bits) {
  int exponent = (bits >> 3) & 15;
  float fraction = float(bits & 7) * 0.125f;
  float value = exponent ? ldexp(1.0f + fraction, exponent - 7) : fraction / 64.0f;
  return bits & 128 ? -value : value;
}

kernel void solve(
    device const uchar* x [[buffer(0)]],
    device const float* xscale [[buffer(1)]],
    device const uchar* weights [[buffer(2)]],
    device const float* wscale [[buffer(3)]],
    device half* output [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= rows * cols)
    return;
  uint r = i / cols, c = i % cols, blocks = width / 128;
  float sum = 0;
  for (uint b = 0; b < blocks; ++b) {
    float dot = 0;
    for (uint j = 0; j < 128; ++j) {
      uint k = b * 128 + j;
      dot += decode(x[r * width + k]) * decode(weights[c * width + k]);
    }
    sum += dot * xscale[r * blocks + b] * wscale[c * blocks + b];
  }
  output[i] = half(sum);
}
