// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const char* a [[buffer(0)]],
    device const char* b [[buffer(1)]],
    device char* c [[buffer(2)]],
    constant uint& m [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant float& scale_a [[buffer(6)]],
    constant float& scale_b [[buffer(7)]],
    constant float& scale_c [[buffer(8)]],
    constant int& zero_a [[buffer(9)]],
    constant int& zero_b [[buffer(10)]],
    constant int& zero_c [[buffer(11)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= m * n)
    return;
  int dot = 0;
  for (uint j = 0; j < k; ++j)
    dot += (int(a[(i / n) * k + j]) - zero_a) * (int(b[j * n + i % n]) - zero_b);
  float value = rint(float(dot) * (scale_a * scale_b / scale_c)) + float(zero_c);
  c[i] = char(clamp(value, -128.0f, 127.0f));
}
