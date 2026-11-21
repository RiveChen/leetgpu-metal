// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device const int* offsets [[buffer(2)]],
    device float* dw [[buffer(3)]],
    constant uint& experts [[buffer(4)]],
    constant uint& tokens [[buffer(5)]],
    constant uint& features [[buffer(6)]],
    constant uint& outputs [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= experts * features * outputs)
    return;
  uint group = i / (features * outputs), k = (i / outputs) % features, n = i % outputs;
  float sum = 0;
  for (int j = offsets[group]; j < offsets[group + 1]; ++j)
    sum += x[j * features + k] * dy[j * outputs + n];
  dw[i] = sum;
}
