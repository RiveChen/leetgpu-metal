// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const half* x [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const half* scale [[buffer(2)]],
    device half* y [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& k [[buffer(6)]],
    constant uint& group [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= m * n)
    return;
  uint row = i / n, col = i % n;
  float sum = 0;
  for (uint j = 0; j < k; ++j) {
    uchar pair = packed[col * (k / 2) + j / 2];
    int value = int((j % 2) ? pair & 15 : pair >> 4) - 8;
    sum += float(x[row * k + j]) * float(value) * float(scale[col * (k / group) + j / group]);
  }
  y[i] = half(sum);
}
