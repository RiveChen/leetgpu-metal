// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const packed_float3* points [[buffer(0)]],
    device int* indices [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  if (i >= n)
    return;
  float best = INFINITY;
  int index = -1;
  for (uint j = 0; j < n; ++j) {
    if (i == j)
      continue;
    float3 difference = float3(points[i]) - float3(points[j]);
    float distance = dot(difference, difference);
    if (distance < best) {
      best = distance;
      index = int(j);
    }
  }
  indices[i] = index;
}
