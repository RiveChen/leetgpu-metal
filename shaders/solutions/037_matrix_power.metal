// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* a [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& power [[buffer(3)]],
    device float* workspace [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= n)
    return;
  device float* row = output + i * n;
  device float* next = workspace + i * n;
  for (uint j = 0; j < n; ++j)
    row[j] = a[i * n + j];
  for (uint step = 1; step < power; ++step) {
    for (uint col = 0; col < n; ++col) {
      float dot = 0;
      for (uint k = 0; k < n; ++k)
        dot += row[k] * a[k * n + col];
      next[col] = dot;
    }
    for (uint j = 0; j < n; ++j)
      row[j] = next[j];
  }
}
