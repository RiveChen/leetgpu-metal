// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* a [[buffer(2)]],
    device const float* b [[buffer(3)]],
    device const int* ids [[buffer(4)]],
    device float* y [[buffer(5)]],
    constant uint& tokens [[buffer(6)]],
    constant uint& features [[buffer(7)]],
    constant uint& outputs [[buffer(8)]],
    constant uint& rank [[buffer(9)]],
    constant uint& adapters [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= tokens * outputs)
    return;
  uint t = i / outputs, col = i % outputs;
  float base = 0;
  for (uint k = 0; k < features; ++k)
    base += x[t * features + k] * weights[k * outputs + col];
  int id = ids[t];
  if (id >= 0) {
    float delta = 0;
    for (uint r = 0; r < rank; ++r) {
      float low = 0;
      for (uint k = 0; k < features; ++k)
        low += x[t * features + k] * a[(id * features + k) * rank + r];
      delta += low * b[(id * rank + r) * outputs + col];
    }
    base += scale * delta;
  }
  y[i] = base;
}
