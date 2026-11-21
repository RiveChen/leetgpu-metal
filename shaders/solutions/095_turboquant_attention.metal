// AI-generated reference solution.
#include <metal_stdlib>
using namespace metal;

kernel void solve(
    device const float* q [[buffer(0)]],
    device const uchar* idx [[buffer(1)]],
    device const char* signs [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* rotation [[buffer(4)]],
    device const float* projection [[buffer(5)]],
    device const float* book [[buffer(6)]],
    device float* out [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    constant uint& length [[buffer(9)]],
    constant uint& width [[buffer(10)]],
    constant uint& levels [[buffer(11)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]
) {

  threadgroup float rotated[256], projected[256];
  for (uint j = lane; j < width; j += 64) {
    float a = 0, b = 0;
    for (uint d = 0; d < width; ++d) {
      float value = q[group * width + d];
      a += rotation[j * width + d] * value;
      b += projection[j * width + d] * value;
    }
    rotated[j] = a;
    projected[j] = b;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint token = lane; token < length; token += 64) {
    float a = 0, b = 0;
    for (uint d = 0; d < width; ++d) {
      uint idx_k = token * width + d;
      a += book[idx[idx_k]] * rotated[d];
      b += float(signs[idx_k]) * projected[d];
    }
    out[group * length + token] = a + b * gamma[token] * 1.2533141373f / float(width);
  }
}
