#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float4* agents [[buffer(0)]],
    device float4* next_agents [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
  // TODO: Update each (x,y,vx,vy) using 5% velocity alignment with other agents strictly within radius 5, then advance its position.
}
