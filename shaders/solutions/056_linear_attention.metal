// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

inline float feature(float x) {
  return x > 0 ? x + 1 : exp(x);
}
kernel void solve(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& length [[buffer(4)]],
    constant uint& width [[buffer(5)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {

  threadgroup float moment[128], mass[128];
  if (lane < width) {
    float a = 0, b = 0;
    for (uint j = 0; j < length; ++j) {
      float phi = feature(k[j * width + lane]);
      a += phi * v[j * width + group];
      b += phi;
    }
    moment[lane] = a;
    mass[lane] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint row = lane; row < length; row += 128) {
    float a = 0, b = 0;
    for (uint d = 0; d < width; ++d) {
      float phi = feature(q[row * width + d]);
      a += phi * moment[d];
      b += phi * mass[d];
    }
    out[row * width + group] = a / b;
  }
}
