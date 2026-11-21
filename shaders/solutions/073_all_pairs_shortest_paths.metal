// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* dist [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& nodes [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]
) {

  for (uint idx = lane; idx < nodes * nodes; idx += 128)
    out[idx] = dist[idx];
  threadgroup_barrier(mem_flags::mem_device);
  for (uint k = 0; k < nodes; ++k) {
    for (uint idx = lane; idx < nodes * nodes; idx += 128) {
      uint row = idx / nodes, col = idx % nodes;
      // With zero diagonal and no negative cycles, the pivot row and column stay fixed.
      if (row != k && col != k)
        out[idx] = min(out[idx], out[row * nodes + k] + out[k * nodes + col]);
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}
