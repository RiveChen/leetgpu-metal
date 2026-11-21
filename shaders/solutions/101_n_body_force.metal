// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* positions [[buffer(0)]],
    device const float* masses [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& particles [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
  float3 origin = float3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]), total = 0;
  for (uint j = 0; j < particles; ++j) {
    float3 r = float3(positions[j * 3], positions[j * 3 + 1], positions[j * 3 + 2]) - origin;
    float inv = rsqrt(dot(r, r) + 1e-6f);
    total += r * (masses[j] * inv * inv * inv);
  }
  for (uint axis = 0; axis < 3; ++axis)
    out[3 * i + axis] = total[axis];
}
