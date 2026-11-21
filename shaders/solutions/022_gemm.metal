// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device half* c [[buffer(0)]],
    device const half* a [[buffer(1)]],
    device const half* b [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    constant float& beta [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= m * n)
    return;
  uint row = i / n, col = i % n;
  float dot = 0;
  for (uint j = 0; j < k; ++j)
    dot += float(a[row * k + j]) * float(b[j * n + col]);
  c[i] = half(alpha * dot + beta * float(c[i]));
}
