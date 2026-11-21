#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const packed_float3* points [[buffer(0)]],
    device int* indices [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: For each 3D point return the closest other point, using the lowest index to break ties.
}
