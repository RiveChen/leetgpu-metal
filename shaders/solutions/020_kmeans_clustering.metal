// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const float* initial_x [[buffer(2)]],
    device const float* initial_y [[buffer(3)]],
    device int* labels [[buffer(4)]],
    device float* center_x [[buffer(5)]],
    device float* center_y [[buffer(6)]],
    constant uint& n [[buffer(7)]],
    constant uint& k [[buffer(8)]],
    constant uint& steps [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
  for (uint c = i; c < k; c += 64) {
    center_x[c] = initial_x[c];
    center_y[c] = initial_y[c];
  }
  for (uint step = 0; step < steps; ++step) {
    threadgroup_barrier(mem_flags::mem_device);
    for (uint j = i; j < n; j += 64) {
      float distance = INFINITY;
      int label = 0;
      for (uint c = 0; c < k; ++c) {
        float2 delta = float2(x[j] - center_x[c], y[j] - center_y[c]);
        float d2 = dot(delta, delta);
        if (d2 < distance) {
          distance = d2;
          label = c;
        }
      }
      labels[j] = label;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint c = i; c < k; c += 64) {
      float2 sum = 0;
      uint count = 0;
      for (uint j = 0; j < n; ++j)
        if (labels[j] == int(c)) {
          sum += float2(x[j], y[j]);
          ++count;
        }
      if (count) {
        center_x[c] = sum.x / count;
        center_y[c] = sum.y / count;
      }
    }
  }
}
