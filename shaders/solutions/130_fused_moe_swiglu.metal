// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const int* ids [[buffer(1)]],
    device const float* mix [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const float* up [[buffer(4)]],
    device const float* down [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& tokens [[buffer(7)]],
    constant uint& hidden [[buffer(8)]],
    constant uint& inner [[buffer(9)]],
    constant uint& experts [[buffer(10)]],
    constant uint& top [[buffer(11)]],
    device float* scratch [[buffer(12)]],
    uint i [[thread_position_in_grid]]
) {
  uint t = i / 64, lane = i % 64;
  device float* state = scratch + t * inner;
  for (uint d = lane; d < hidden; d += 64)
    output[t * hidden + d] = 0;
  for (uint s = 0; s < top; ++s) {
    uint e = ids[t * top + s];
    for (uint k = lane; k < inner; k += 64) {
      float a = 0, b = 0;
      for (uint j = 0; j < hidden; ++j) {
        uint idx = (e * hidden + j) * inner + k;
        a += x[t * hidden + j] * gate[idx];
        b += x[t * hidden + j] * up[idx];
      }
      state[k] = (a / (1 + exp(-a))) * b;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint d = lane; d < hidden; d += 64) {
      float val = 0;
      for (uint k = 0; k < inner; ++k)
        val += state[k] * down[(e * inner + k) * hidden + d];
      output[t * hidden + d] += mix[t * top + s] * val;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}
