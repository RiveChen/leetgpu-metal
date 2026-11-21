// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const int* flags [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= n || !flags[i])
    return;
  float sum = 0;
  for (uint j = i; j < n && (j == i || !flags[j]); ++j) {
    y[j] = sum;
    sum += x[j];
  }
}
