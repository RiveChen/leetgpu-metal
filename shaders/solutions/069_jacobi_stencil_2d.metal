// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device float* y [[buffer(1)]],
    constant uint& h [[buffer(2)]],
    constant uint& w [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= h * w)
    return;
  uint row = i / w, col = i % w;
  y[i] = (row == 0 || col == 0 || row + 1 == h || col + 1 == w)
             ? x[i]
             : .25f * (x[i - w] + x[i + w] + x[i - 1] + x[i + 1]);
}
