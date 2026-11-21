// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const char* weights [[buffer(2)]],
    device const float* scales [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant uint& width [[buffer(7)]],
    uint index [[thread_position_in_grid]]
) {

  if (index >= rows * cols)
    return;
  uint row = index / cols, col = index % cols;
  float sq = 0, peak = 0;
  for (uint j = 0; j < width; ++j) {
    float v = x[row * width + j];
    sq += v * v;
    peak = max(peak, abs(v * gamma[j]));
  }
  if (peak == 0) {
    out[index] = 0;
    return;
  }
  float step = peak / 127, norm = rsqrt(sq / width + 1e-6f);
  int product = 0;
  for (uint j = 0; j < width; ++j) {
    int q = int(clamp(rint(x[row * width + j] * gamma[j] / step), -127.f, 127.f));
    product += q * int(weights[col * width + j]);
  }
  out[index] = float(product) * step * norm * scales[col];
}
