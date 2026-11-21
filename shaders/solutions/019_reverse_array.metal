// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device float* x [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
  if (i < n / 2) {
    float value = x[i];
    x[i] = x[n - 1 - i];
    x[n - 1 - i] = value;
  }
}
