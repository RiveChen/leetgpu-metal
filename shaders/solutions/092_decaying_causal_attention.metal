// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    constant float& decay [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
  uint row = i / width, col = i % width;
  float sum = 0, weight = 1;
  for (int j = int(row); j >= 0; --j) {
    float dot = 0;
    for (uint d = 0; d < width; ++d)
      dot += q[row * width + d] * k[j * width + d];
    sum += weight * dot * v[j * width + col];
    weight *= decay;
  }
  out[i] = sum * rsqrt(float(width));
}
