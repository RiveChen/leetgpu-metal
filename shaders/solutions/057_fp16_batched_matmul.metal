// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= batch * m * n)
    return;
  uint group = i / (m * n), row = (i / n) % m, col = i % n;
  float sum = 0;
  for (uint j = 0; j < k; ++j)
    sum += float(a[(group * m + row) * k + j]) * float(b[(group * k + j) * n + col]);
  output[i] = half(sum);
}
