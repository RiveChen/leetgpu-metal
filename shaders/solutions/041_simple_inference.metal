// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& features [[buffer(5)]],
    constant uint& outputs [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch * outputs)
    return;
  uint row = i / outputs, col = i % outputs;
  float sum = bias[col];
  for (uint k = 0; k < features; ++k)
    sum += weights[col * features + k] * x[row * features + k];
  y[i] = sum;
}
