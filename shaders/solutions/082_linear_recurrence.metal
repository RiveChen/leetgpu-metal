// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* h [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch)
    return;
  float state = 0;
  for (uint t = 0; t < length; ++t) {
    uint offset = i * length + t;
    state = a[offset] * state + x[offset];
    h[offset] = state;
  }
}
