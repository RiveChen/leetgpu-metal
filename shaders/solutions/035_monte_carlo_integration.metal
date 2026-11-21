// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* y [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant float& a [[buffer(2)]],
    constant float& b [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
  float sum = 0;
  for (uint j = 0; j < n; ++j)
    sum += y[j];
  result[0] = (b - a) * (sum / float(n));
}
