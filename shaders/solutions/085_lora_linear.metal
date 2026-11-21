// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device const float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device float* y [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    constant uint& inputs [[buffer(6)]],
    constant uint& outputs [[buffer(7)]],
    constant uint& rank [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch * outputs)
    return;
  uint row = i / outputs, col = i % outputs;
  float base = 0, delta = 0;
  for (uint j = 0; j < inputs; ++j)
    base += x[row * inputs + j] * w[col * inputs + j];
  for (uint r = 0; r < rank; ++r) {
    float hidden = 0;
    for (uint j = 0; j < inputs; ++j)
      hidden += x[row * inputs + j] * a[r * inputs + j];
    delta += hidden * b[col * rank + r];
  }
  y[i] = base + scale * delta;
}
